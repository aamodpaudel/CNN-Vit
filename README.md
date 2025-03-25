# CNN-Vit
# Pet Sentiment Classifier (CNN-ViT Hybrid Model):  A deep learning project to classify pet emotions (angry, happy, relaxed, sad) using a Vision Transformer (ViT) model with PyTorch and Streamlit.


## 📋 Project Overview
- **Objective**: Balance and classify pet emotion images using data augmentation and a ViT-CNN hybrid model.
- **Dataset**: 7,400 raw images across 4 classes (angry, happy, sad, relaxed).
- **Augmentation**: Balances classes to ~2,000 images each using flips, rotations, and crops.
- **Model**: Fine-tuned pretrained ViT-b-16 model for classification.
- **UI**: Streamlit web app for real-time predictions.

## 🛠 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/aamodpaudel/CNN-Vit
   cd CNN-Vit
