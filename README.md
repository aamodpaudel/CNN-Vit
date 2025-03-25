# CNN-Vit
# Pet Sentiment Classifier (CNN-ViT Hybrid Model):  A deep learning project to classify pet emotions (angry, happy, relaxed, sad) using a Vision Transformer (ViT) model with PyTorch and Streamlit.


## 📋 Project Overview
- **Objective**: Balance and classify pet emotion images using data augmentation and a ViT-CNN hybrid model.
- **Dataset**: 7,400 raw images across 4 classes (angry, happy, sad, relaxed).
- **Augmentation**: Balances classes to ~2,000 images each using flips, rotations, and crops.
- **Model**: In the hybrid CNN-ViT model, we did not use a standard pre-defined ViT model (like vit_b_16). Instead, we built a custom transformer encoder on top of the CNN backbone.
- **UI**: Streamlit web app for real-time predictions.

- **Before Augmentation**: The subfolders had the following files size - angry had 988,
happy had 4355, relaxed had 1203, sad had 844.
-----------------------------------------------------
**Hybrid Architecture Components**
A. CNN Backbone: ResNet-18

Extracts spatial features from the input image.

Output shape: [batch, 512, 7, 7].

B. Custom ViT-like Components:

Feature Projection:
A 1x1 convolutional layer to project ResNet-18 features into a transformer-based model.

CLS Token:
A token (nn.Parameter) added to the feature sequence (similar to ViT’s [CLS] token).

Positional Embeddings:
Added positional embeddings for global context.

Transformer Encoder:
A lightweight transformer encoder to process the combined features.

--------------------------------------------------------------
## 🛠 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/aamodpaudel/CNN-Vit
   cd CNN-Vit
