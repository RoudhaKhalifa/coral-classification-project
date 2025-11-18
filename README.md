# Coral Classification with CBAM-Enhanced ResNet18

A deep learning project for classifying healthy vs bleached corals using ResNet18 with Convolutional Block Attention Module (CBAM) and CLAHE preprocessing.

## Project Overview

This project implements a computer vision model to distinguish between healthy and bleached coral reefs, which is crucial for monitoring ocean health and climate change impacts. The model combines:

- **ResNet18** backbone with ImageNet pre-trained weights
- **CBAM (Convolutional Block Attention Module)** for enhanced feature attention
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)** preprocessing
- **Grad-CAM** visualization for model interpretability

## Features

- Binary classification: Healthy vs Bleached corals
- Channel and spatial attention mechanisms via CBAM
- Enhanced image preprocessing with CLAHE
- Comprehensive data augmentation
- Model interpretability through Grad-CAM visualizations
- Detailed performance metrics and confusion matrix

## Requirements

```bash
torch
torchvision
torchcam
opencv-python
numpy
matplotlib
scikit-learn
Pillow
tqdm
kagglehub
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/RoudhaKhalifa/coral-classification-project.git
cd coral-classification-project
```

2. Install dependencies:
```bash
pip install torch torchvision torchcam opencv-python numpy matplotlib scikit-learn Pillow tqdm kagglehub
```

## Dataset

The project uses the [Corals Classification Dataset](https://www.kaggle.com/datasets/aneeshdighe/corals-classification) from Kaggle.

The dataset is automatically downloaded using:
```python
import kagglehub
path = kagglehub.dataset_download("aneeshdighe/corals-classification")
```

**Dataset Structure:**
- Training set: Images of healthy and bleached corals
- Testing set: Separate test images for validation
- Two classes: `healthy_corals` and `bleached_corals`

## 🏗️ Model Architecture

### CBAM-ResNet18
The model integrates CBAM attention modules into ResNet18's layer4:

1. **Channel Attention**: Focuses on "what" is meaningful
   - Uses both average and max pooling
   - Shared MLP with reduction ratio of 16

2. **Spatial Attention**: Focuses on "where" is meaningful
   - Applies 7×7 convolution on concatenated avg/max features
   - Generates spatial attention map

3. **ResNet18 Backbone**: Pre-trained on ImageNet, modified final layer for binary classification

### Preprocessing Pipeline

**Training Augmentation:**
- CLAHE enhancement for better contrast
- Random horizontal flip (50%)
- Random vertical flip (30%)
- Color jitter (brightness, contrast, saturation, hue)
- Random rotation (±15°)
- Resize to 224×224

**Test Transform:**
- CLAHE enhancement
- Resize to 224×224

## Usage

### Training the Model

```python
# The model trains for 5 epochs with Adam optimizer
# Learning rate: 1e-4
# Batch size: 8
# Loss function: CrossEntropyLoss

python coral_classification.py
```

### Key Training Parameters:
- Epochs: 5 (adjustable)
- Optimizer: Adam (lr=1e-4)
- Batch size: 8
- Device: CUDA if available, else CPU

### Model Evaluation

The script automatically evaluates the model and generates:
- Classification report (precision, recall, F1-score)
- Confusion matrix visualization
- Grad-CAM visualizations for interpretability

### Loading Pre-trained Model

```python
model = ResNetCBAM(num_classes=2).to(device)
model.load_state_dict(torch.load("resnet18_cbam_coral_clahe.pth"))
model.eval()
```

## 📈 Results

The model generates comprehensive evaluation metrics:

- **Training/Validation Loss and Accuracy** per epoch
- **Classification Report** with precision, recall, and F1-scores
- **Confusion Matrix** showing true vs predicted classifications
- **Grad-CAM Visualizations** highlighting regions the model focuses on

## Grad-CAM Visualization

The project includes Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize which parts of the coral images the model focuses on when making predictions:

```python
from torchcam.methods import GradCAM
cam_extractor = GradCAM(model, target_layer="backbone.layer4")
```

Visualizations show:
- Original images (healthy and bleached corals)
- Heatmap overlays indicating attention regions
- Predicted class labels

## Project Structure

```
coral-classification-project/
│
├── coral_classification.py    # Main training and evaluation script
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
└── resnet18_cbam_coral_clahe.pth  # Saved model weights (after training)
```

## Key Concepts

### CBAM (Convolutional Block Attention Module)
Enhances CNN feature maps through sequential channel and spatial attention, allowing the network to focus on important features and regions.

### CLAHE (Contrast Limited Adaptive Histogram Equalization)
Improves local contrast in images, particularly useful for underwater coral images with varying lighting conditions.

### Transfer Learning
Uses ImageNet pre-trained ResNet18 as a starting point, leveraging learned features for coral classification.

## Customization

### Adjust Training Parameters:
```python
epochs = 10  # Increase for better convergence
optimizer = optim.Adam(model.parameters(), lr=5e-5)  # Lower learning rate
batch_size = 16  # Larger batch size if GPU memory allows
```

### Modify CBAM Attention:
```python
# Change reduction ratio in ChannelAttention
self.ca = ChannelAttention(planes, ratio=8)  # More parameters

# Change kernel size in SpatialAttention
self.sa = SpatialAttention(kernel_size=5)  # Smaller receptive field
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{coral_classification_cbam,
  author = {Roudha},
  title = {Coral Classification with CBAM-Enhanced ResNet18},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/RoudhaKhalifa/coral-classification-project}
}
```


## Acknowledgments

- Dataset: [Aneesh Dighe's Corals Classification Dataset](https://www.kaggle.com/datasets/aneeshdighe/corals-classification)
- CBAM Paper: [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
- PyTorch and torchvision teams for the excellent deep learning framework
- torchcam for Grad-CAM implementation

---

**Note**: This project was developed in Google Colab and can be easily adapted for local execution or other cloud platforms.
