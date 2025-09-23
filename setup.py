"""
Setup script for Medical AI - Colon Polyp Segmentation
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="medical-ai-polyp-segmentation",
    version="1.0.0",
    author="Medical AI Research Team",
    author_email="research@medical-ai.org",
    description="AI-powered colon polyp segmentation using FlexibleUNet and EfficientNet-B4",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/medical-ai/Hydra-MONAI-Lightning-FlexibleUNet-ColonPolypSegmentation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA",
        "Environment :: Web Environment",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "sphinxcontrib-napoleon>=0.7",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "plotly>=5.0.0",
            "seaborn>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "medical-ai-train=train:main",
            "medical-ai-web=app:main",
            "medical-ai-desktop=desktop_app:main",
            "medical-ai-test=yarisma_test:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md", "*.txt"],
        "config": ["**/*.yaml", "**/*.yml"],
    },
    keywords=[
        "medical-ai",
        "polyp-segmentation",
        "colonoscopy",
        "deep-learning",
        "computer-vision",
        "medical-imaging",
        "pytorch-lightning",
        "monai",
        "unet",
        "efficientnet",
        "healthcare",
        "artificial-intelligence",
        "machine-learning",
        "medical-diagnosis",
        "image-segmentation",
    ],
    project_urls={
        "Bug Reports": "https://github.com/medical-ai/Hydra-MONAI-Lightning-FlexibleUNet-ColonPolypSegmentation/issues",
        "Source": "https://github.com/medical-ai/Hydra-MONAI-Lightning-FlexibleUNet-ColonPolypSegmentation",
        "Documentation": "https://medical-ai.github.io/Hydra-MONAI-Lightning-FlexibleUNet-ColonPolypSegmentation/",
        "Changelog": "https://github.com/medical-ai/Hydra-MONAI-Lightning-FlexibleUNet-ColonPolypSegmentation/blob/main/CHANGELOG.md",
        "Funding": "https://github.com/sponsors/medical-ai",
    },
    zip_safe=False,
    
    # Medical AI specific metadata
    medical_ai_info={
        "purpose": "Research and Educational Use Only",
        "validation_dice_score": 0.854,
        "model_architecture": "FlexibleUNet + EfficientNet-B4",
        "dataset": "Kvasir-SEG",
        "input_resolution": "320x320",
        "clinical_validation": "Not for clinical diagnosis",
        "regulatory_status": "Research Use Only",
    },
)