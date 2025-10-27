# Medical-Image-Classifier

This project is a **deep learning image classification system** that uses a **Vision Transformer (ViT)** model to automatically detect **Pneumonia** from **chest X-ray images**.  
Built with **PyTorch** and **Hugging Face Transformers**, it demonstrates how modern transformer architectures can outperform traditional CNNs in medical imaging tasks.  

---

## Overview  

The model classifies X-ray images into two categories:  
- **Normal**  
- **Pneumonia**  

It was fine-tuned from the pretrained **`google/vit-base-patch16-224-in21k`** Vision Transformer model.  
The repository includes a ready-to-run inference script that takes an image and outputs the predicted label with a confidence score.

---

## Key Features  

✅ Uses **Vision Transformer (ViT)** for image understanding  
✅ Built on **Hugging Face Transformers** and **PyTorch**  
✅ Handles **medical image classification** efficiently  
✅ Prints clear predictions with **confidence levels**  
✅ Easily extensible for other medical or image datasets  

---

## Project Structure  

vit-medical-image-classifier/

│

├── vit_xray_final/                 # Saved fine-tuned model

├── datasets/

│   └── chest_xray/

│       ├── train/

│       ├── test/

│       └── val/

├── inference.py                    # Inference / prediction script

├── image_classifier.py                        # Training script

└── README.md                       # Project documentation

## Example Output

Loading model...
Running inference on: datasets/chest_xray/test/PNEUMONIA/person1680_virus_2897.jpeg

Prediction: Pneumonia
Confidence: 97.32%

✅ Inference complete.

## Author

Sharvari Sunil Pradhan

AI/ML Enthusiast | BSc Computer Science
