from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# Load trained model and processor
MODEL_PATH = "./vit_xray_final"

print("Loading model...")
model = ViTForImageClassification.from_pretrained(MODEL_PATH)
processor = ViTImageProcessor.from_pretrained(MODEL_PATH)

# Choose test image
image_path = "datasets/chest_xray/test/PNEUMONIA/person1680_virus_2897.jpeg"

print(f"Running inference on: {image_path}")

# Load and preprocess the image
image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")

# Get model prediction
with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()

# Display result
label = "Pneumonia" if prediction == 1 else "Normal"
print(f"\nPrediction: {label}")

# Confidence score
probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
confidence = probs[0][prediction].item() * 100
print(f"Confidence: {confidence:.2f}%")

print("\n✅ Inference complete.")

# python inference.py