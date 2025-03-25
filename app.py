import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import torchvision

# 1. Define the ViTModel class (MUST match training code)
class ViTModel(torch.nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.vit = torchvision.models.vit_b_16(pretrained=False)  # Load from torchvision
        self.vit.heads.head = torch.nn.Linear(self.vit.heads.head.in_features, num_classes)
    
    def forward(self, x):
        return self.vit(x)

# 2. Load the trained weights
@st.cache_resource
def load_model():
    model = ViTModel()
    model.load_state_dict(torch.load("pet_sentiment_model.pth", map_location='cpu'))
    model.eval()
    return model

model = load_model()

# 3. Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Streamlit UI
st.title("Pet Sentiment Predictor")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output).item()
    
    st.write(f"Prediction: {['angry', 'happy', 'relaxed', 'sad'][prediction]}")