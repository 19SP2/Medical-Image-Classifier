from transformers import ViTForImageClassification, ViTImageProcessor, Trainer, TrainingArguments
import torch
from PIL import Image
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import pandas as pd

# Load model and processor
model_name = "google/vit-base-patch16-224-in21k"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=2,  # 0 = Normal, 1 = Pneumonia
    ignore_mismatched_sizes=True
)

# Load image paths and labels into a DataFrame
data = []
for label in ["NORMAL", "PNEUMONIA"]:
    folder = f"datasets/chest_xray/train/{label}"
    for file in os.listdir(folder):
        if file.endswith((".png", ".jpeg", ".jpg")):
            data.append([os.path.join(folder, file), label])

df = pd.DataFrame(data, columns=["file_path", "label"])

# Convert labels to 0 (Normal) / 1 (Pneumonia)
y = df["label"].apply(lambda x: 0 if x.upper() == "NORMAL" else 1).values
X = df["file_path"].values

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Custom Dataset for Trainer
class XRayDataset(Dataset):
    def __init__(self, file_paths, labels, processor):
        self.file_paths = file_paths
        self.labels = labels
        self.processor = processor # ViTImageProcessor

    def __len__(self):
        return len(self.file_paths) # Returns the number of images in the dataset

    def __getitem__(self, idx):
        image = Image.open(self.file_paths[idx]).convert("RGB")
        label = self.labels[idx]
        encoding = self.processor(images=image, return_tensors="pt") # Converts the PIL image into a pytorch tensor
        # Remove batch dimension
        item = {k: v.squeeze() for k, v in encoding.items()}
        item["labels"] = torch.tensor(label)
        return item

train_dataset = XRayDataset(X_train, y_train, processor)
eval_dataset = XRayDataset(X_test, y_test, processor)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./vit_xray_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=2e-5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Train
print("Starting training...")
trainer.train()

# Evaluate
print("\nEvaluating on test set...")
results = trainer.evaluate()
print(f"Test Accuracy: {results['eval_accuracy']:.4f}")

# Save model & processor
model.save_pretrained("./vit_xray_final")
processor.save_pretrained("./vit_xray_final")
print("Model saved!")

# Test inference with saved model
print("\n" + "="*50)
print("Testing saved model:")

# Load saved model and processor
from transformers import ViTForImageClassification, ViTImageProcessor

model = ViTForImageClassification.from_pretrained("./vit_xray_final")
processor = ViTImageProcessor.from_pretrained("./vit_xray_final")

# Test image
test_image = Image.open("datasets/chest_xray/test/PNEUMONIA/person1680_virus_2897.jpeg").convert("RGB")
inputs = processor(images=test_image, return_tensors="pt")
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=-1).item()
print(f"Prediction: {'Pneumonia' if prediction == 1 else 'Normal'}")

# python Image_Classifier.py