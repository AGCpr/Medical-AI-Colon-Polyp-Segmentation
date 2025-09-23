# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-22

### Added
- **Core Features**
  - FlexibleUNet model with EfficientNet-B4 backbone for polyp segmentation
  - PyTorch Lightning integration for streamlined training
  - Hydra configuration management system
  - MONAI framework integration for medical image processing

- **Applications**
  - Desktop GUI application using Tkinter for real-time inference
  - Web-based interface using Gradio for browser-based analysis
  - Competition evaluation framework with IoU metrics
  - Medical-grade performance assessment tools

- **Configuration System**
  - Modular YAML configuration files
  - Hydra-based parameter management
  - WandB hyperparameter sweeping support
  - Structured experiment tracking

- **Dataset Support**
  - Kvasir-SEG dataset integration
  - Automatic train/validation/test splitting
  - MONAI transforms for data augmentation
  - Flexible dataset loading pipeline

- **Model Training**
  - DiceLoss optimization for segmentation
  - EarlyStopping and ModelCheckpoint callbacks
  - Learning rate scheduling with ReduceLROnPlateau
  - Comprehensive logging and monitoring

- **Evaluation & Testing**
  - Competition-compliant IoU calculation
  - Dice metric evaluation
  - Automated performance grading
  - Batch evaluation capabilities

- **Visualization**
  - Segmentation mask overlays
  - Confidence heatmaps
  - Training progress plots
  - Real-time prediction display

### Performance
- **Model Metrics**
  - Validation Dice Score: 0.854
  - Competition Grade: MÜKEMMEL
  - Input Resolution: 320×320 pixels
  - Inference Speed: ~50ms per image (GPU)

### Technical Details
- **Architecture**: MONAI FlexibleUNet + EfficientNet-B4
- **Framework**: PyTorch Lightning 2.0+
- **Configuration**: Hydra Core 1.3+
- **Medical AI**: MONAI 1.3+
- **Web Interface**: Gradio 4.44+
- **Experiment Tracking**: WandB integration

### Documentation
- Comprehensive README with installation guide
- Configuration file documentation
- API documentation for all modules
- Clinical usage guidelines
- Performance benchmarking results

## [0.1.0] - Initial Development

### Added
- Initial project structure
- Basic model implementation
- Training pipeline setup
- Configuration system foundation