import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18  
import json


class HybridCNNViT(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        
        # CNN Backbone (ResNet-18)
        self.cnn = resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2]) 
        
       
        self.feature_projection = nn.Conv2d(
            in_channels=512,  # ResNet-18's last layers
            out_channels=768,  # ViT's hidden new layers
            kernel_size=1
        )
        
        # ViT Components
        self.cls_token = nn.Parameter(torch.randn(1, 1, 768))
        self.positional_embedding = nn.Parameter(torch.randn(1, 50, 768))  
        
        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=768,
                nhead=8,
                dim_feedforward=3072,
                activation="gelu"
            ),
            num_layers=4
        )
        
        
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        
        features = self.cnn(x)  
        
        
        features = self.feature_projection(features)  
        features = features.flatten(2).permute(0, 2, 1)  
        
      
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        features = torch.cat((cls_tokens, features), dim=1) 
        
        
        features += self.positional_embedding
        
        
        features = features.permute(1, 0, 2)  
        features = self.transformer(features)
        
       
        cls_output = features[0]  
        return self.classifier(cls_output)


data_dir = r"D:\College\augmented_data"
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridCNNViT(num_classes=4).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
metrics = []








for epoch in range(10):
  

   
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    
    avg_train_loss = train_loss / len(train_dataset)
    
   
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_val_loss = val_loss / len(val_dataset)
    val_accuracy = correct / total
    
   
    epoch_metrics = {
        'epoch': epoch + 1,
        'train_loss': float(avg_train_loss),
        'val_loss': float(avg_val_loss),
        'val_accuracy': float(val_accuracy)
    }
    metrics.append(epoch_metrics)
    
    print(f"Epoch {epoch+1}")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss: {avg_val_loss:.4f}")
    print(f"  Val Accuracy: {val_accuracy:.4f}\n")


with open('training_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

torch.save(model.state_dict(), "pet_sentiment_model.pth")