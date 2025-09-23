import torch
import pytorch_lightning as pl
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.networks.nets import FlexibleUNet
from hydra.utils import instantiate
from omegaconf import DictConfig


class SegmentationModel(pl.LightningModule):
    def __init__(
        self, model_cfg: DictConfig, optimizer_cfg: dict, scheduler_cfg: dict = None
    ):

        super().__init__()
        self.save_hyperparameters()

        self.model_cfg = model_cfg
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        # Model
        self.model = FlexibleUNet(
            in_channels=model_cfg.in_channels,
            out_channels=model_cfg.out_channels,
            backbone=model_cfg.backbone,
            pretrained=model_cfg.pre_trained,
            decoder_channels=model_cfg.decoder_channels,
            spatial_dims=model_cfg.spatial_dimensions,
        )

        # Loss function
        if hasattr(model_cfg, "loss"):
            self.loss_fn = instantiate(model_cfg.loss)
        else:
            self.loss_fn = DiceLoss(sigmoid=True)

        # Metrics
        if hasattr(model_cfg, "metrics") and hasattr(model_cfg.metrics, "dice"):
            self.train_dice = instantiate(model_cfg.metrics.dice)
            self.val_dice = instantiate(model_cfg.metrics.dice)
            self.test_dice = instantiate(model_cfg.metrics.dice)
        else:
            self.train_dice = DiceMetric(
                include_background=True, reduction="mean", get_not_nans=False
            )
            self.val_dice = DiceMetric(
                include_background=True, reduction="mean", get_not_nans=False
            )
            self.test_dice = DiceMetric(
                include_background=True, reduction="mean", get_not_nans=False
            )

        # For storing test predictions
        self.test_predictions = {"inputs": [], "preds": [], "targets": []}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_res = self(x)
        loss = self.loss_fn(y_res, y)

        preds = (torch.sigmoid(y_res) > 0.5).float()
        y = y.float()  # Ensuring label is float tensor(fix for an error)

        # debugging
        if preds.shape != y.shape:
            print(f"Mismatch: preds {preds.shape}, labels {y.shape}")

        self.train_dice(preds, y)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_dice", self.train_dice.aggregate().item(), prog_bar=True)
        self.train_dice.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_res = self(x)
        loss = self.loss_fn(y_res, y)
        preds = (torch.sigmoid(y_res) > 0.5).float()
        self.val_dice(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_dice", self.val_dice.aggregate().item(), prog_bar=True)
        self.val_dice.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_res = self(x)
        loss = self.loss_fn(y_res, y)
        preds = (torch.sigmoid(y_res) > 0.5).float()

        self.test_dice(preds, y)

        # Accumulate predictions
        self.test_predictions["inputs"].append(x.cpu())
        self.test_predictions["preds"].append(preds.cpu())
        self.test_predictions["targets"].append(y.cpu())

    def on_test_epoch_end(self):
        self.log("test_dice", self.test_dice.aggregate().item(), prog_bar=True)
        self.test_dice.reset()

        # Stack stored predictions
        self.test_predictions["inputs"] = torch.cat(
            self.test_predictions["inputs"], dim=0
        )
        self.test_predictions["preds"] = torch.cat(
            self.test_predictions["preds"], dim=0
        )
        self.test_predictions["targets"] = torch.cat(
            self.test_predictions["targets"], dim=0
        )

    def configure_optimizers(self):
        # Instantiate optimizer with model parameters
        optimizer_cfg = self.optimizer_cfg.copy()
        optimizer_class = optimizer_cfg.pop("_target_")

        if optimizer_class in ["torch.optim.Adam", "torch.optim.AdamW"]:
            optimizer_cfg.pop("momentum", None)

        # Get the optimizer class
        if optimizer_class == "torch.optim.Adam":
            optimizer = torch.optim.Adam(self.parameters(), **optimizer_cfg)
        elif optimizer_class == "torch.optim.SGD":
            optimizer = torch.optim.SGD(self.parameters(), **optimizer_cfg)
        elif optimizer_class == "torch.optim.AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), **optimizer_cfg)
        else:
            # Use instantiate for other optimizers
            optimizer = instantiate(
                {"_target_": optimizer_class, **optimizer_cfg}, params=self.parameters()
            )

        # Configure scheduler if provided
        if self.scheduler_cfg:
            scheduler = instantiate(self.scheduler_cfg, optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_dice",
                    "frequency": 1,
                },
            }

        return optimizer
