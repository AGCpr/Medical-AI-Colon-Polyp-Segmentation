import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import torch

from model import SegmentationModel


def default_ckpt_path() -> str:
    logs_dir = os.path.join(os.getcwd(), "lightning_logs")
    if not os.path.isdir(logs_dir):
        return ""
    versions = [d for d in os.listdir(logs_dir) if d.startswith("version_")]
    if not versions:
        return ""
    versions_sorted = sorted(versions, key=lambda s: int(s.split("_")[-1]))
    latest = os.path.join(logs_dir, versions_sorted[-1], "checkpoints", "best_model.ckpt")
    return latest if os.path.exists(latest) else ""


class DesktopApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Colon Polyp Segmentation - Offline")
        self.root.geometry("1100x700")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_input_size = (320, 320)
        self.ckpt_path = tk.StringVar(value=default_ckpt_path())
        self.image_path = tk.StringVar(value="")
        self.threshold = tk.DoubleVar(value=0.5)

        self._build_ui()

    def _build_ui(self):
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(top_frame, text="Checkpoint:").pack(side=tk.LEFT)
        self.ckpt_entry = ttk.Entry(top_frame, textvariable=self.ckpt_path, width=80)
        self.ckpt_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Browse", command=self._browse_ckpt).pack(side=tk.LEFT)
        ttk.Button(top_frame, text="Load Model", command=self._load_model).pack(side=tk.LEFT, padx=10)

        img_frame = ttk.Frame(self.root)
        img_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(img_frame, text="Input Image:").pack(side=tk.LEFT)
        self.img_entry = ttk.Entry(img_frame, textvariable=self.image_path, width=80)
        self.img_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(img_frame, text="Browse", command=self._browse_image).pack(side=tk.LEFT)

        controls = ttk.Frame(self.root)
        controls.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(controls, text="Threshold:").pack(side=tk.LEFT)
        self.thr_scale = ttk.Scale(controls, from_=0.0, to=1.0, variable=self.threshold, orient=tk.HORIZONTAL)
        self.thr_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        ttk.Button(controls, text="Predict", command=self._predict).pack(side=tk.LEFT, padx=10)

        views = ttk.Frame(self.root)
        views.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas_input = tk.Label(views)
        self.canvas_input.pack(side=tk.LEFT, expand=True, padx=5)

        right_views = ttk.Frame(views)
        right_views.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.canvas_mask = tk.Label(right_views)
        self.canvas_mask.pack(side=tk.TOP, expand=True, pady=5)

        self.canvas_overlay = tk.Label(right_views)
        self.canvas_overlay.pack(side=tk.TOP, expand=True, pady=5)

        self.status_var = tk.StringVar(value="Model not loaded.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _browse_ckpt(self):
        path = filedialog.askopenfilename(title="Select checkpoint", filetypes=[("Checkpoint", "*.ckpt"), ("All files", "*.*")])
        if path:
            self.ckpt_path.set(path)

    def _browse_image(self):
        path = filedialog.askopenfilename(title="Select image", filetypes=[("Images", "*.png;*.jpg;*.jpeg"), ("All files", "*.*")])
        if path:
            self.image_path.set(path)
            self._show_image(path, self.canvas_input, max_size=(512, 512))

    def _load_model(self):
        ckpt = self.ckpt_path.get().strip()
        if not ckpt or not os.path.exists(ckpt):
            messagebox.showerror("Error", f"Checkpoint not found: {ckpt}")
            return
        try:
            self.model = SegmentationModel.load_from_checkpoint(ckpt)
            self.model.eval()
            self.model.to(self.device)

            # Extract input size from model hparams
            if hasattr(self.model, "hparams") and "spatial_size" in self.model.hparams:
                self.model_input_size = tuple(self.model.hparams["spatial_size"])
            else:
                # Fallback to default if not found
                self.model_input_size = (320, 320)
                messagebox.showwarning("Warning", "Could not determine model input size from checkpoint. Falling back to default (320, 320).")

            self.status_var.set(f"Model loaded: {os.path.basename(ckpt)} on {self.device} | Input size: {self.model_input_size}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")

    def _preprocess(self, img: Image.Image) -> torch.Tensor:
        size = self.model_input_size
        img_resized = img.convert("RGB").resize(size)
        arr = np.array(img_resized).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # CHW
        return torch.from_numpy(arr)[None, ...]

    def _predict(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Load the model first.")
            return
        img_path = self.image_path.get().strip()
        if not img_path or not os.path.exists(img_path):
            messagebox.showwarning("Warning", "Select a valid image.")
            return

        try:
            img = Image.open(img_path)
            x = self._preprocess(img).to(self.device)
            with torch.no_grad():
                y = self.model(x)
                prob = torch.sigmoid(y)[0, 0].cpu().numpy()
                mask = (prob > float(self.threshold.get())).astype(np.uint8) * 255

            # Prepare visuals
            mask_img = Image.fromarray(mask)
            base = img.resize(mask_img.size).convert("RGB")
            overlay = np.array(base)
            red = np.zeros_like(overlay)
            red[..., 0] = 255
            alpha = 0.35
            overlay = (overlay * (1 - alpha) + red * alpha * (mask[..., None] / 255.0)).astype(np.uint8)
            overlay_img = Image.fromarray(overlay)

            self._show_image(mask_img, self.canvas_mask, max_size=(512, 256))
            self._show_image(overlay_img, self.canvas_overlay, max_size=(512, 256))
            self.status_var.set("Prediction done.")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")

    @staticmethod
    def _show_image(img_or_path, widget: tk.Label, max_size=(512, 512)):
        if isinstance(img_or_path, (str, os.PathLike)):
            img = Image.open(img_or_path)
        else:
            img = img_or_path
        img = img.copy()
        img.thumbnail(max_size)
        tk_img = ImageTk.PhotoImage(img)
        widget.configure(image=tk_img)
        widget.image = tk_img  # prevent GC


if __name__ == "__main__":
    root = tk.Tk()
    app = DesktopApp(root)
    root.mainloop()


