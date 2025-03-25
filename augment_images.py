import os
from PIL import Image
import shutil
import random
from torchvision import transforms


augmentation_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
])

def augment_class(source_dir, target_dir, class_name, target_count=2000):
    class_source = os.path.join(source_dir, class_name)
    class_target = os.path.join(target_dir, class_name)
    os.makedirs(class_target, exist_ok=True)

    image_files = [f for f in os.listdir(class_source) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    original_count = len(image_files)

    if original_count < target_count:
        required_augmented = target_count - original_count
        aug_per_image = required_augmented // original_count
        remaining = required_augmented % original_count

        
        for img_file in image_files:
            shutil.copy(os.path.join(class_source, img_file), class_target)

   
        for i, img_file in enumerate(image_files):
            img = Image.open(os.path.join(class_source, img_file)).convert("RGB")  # Ensure image is in RGB mode
            for j in range(aug_per_image):
                augmented_img = augmentation_transform(img)
                augmented_img.save(os.path.join(class_target, f"aug_{i}_{j}.jpg"))
            if i < remaining:
                augmented_img = augmentation_transform(img)
                augmented_img.save(os.path.join(class_target, f"aug_{i}_extra.jpg"))
            img.close()
    else:
        selected_files = random.sample(image_files, target_count)
        for img_file in selected_files:
            shutil.copy(os.path.join(class_source, img_file), os.path.join(class_target, img_file))

def main_augmentation():
    source_dir = r"D:\College\raw_datasets"
    target_dir = r"D:\College\augmented_data"
    classes = ["angry", "happy", "sad", "relaxed"]
    for cls in classes:
        augment_class(source_dir, target_dir, cls)

if __name__ == "__main__":
    main_augmentation()