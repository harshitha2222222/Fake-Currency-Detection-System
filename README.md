


# Fake Currency Detection System

A Computer Vision-based system that detects counterfeit currency using OpenCV and machine learning/deep learning models. It identifies security features such as texture, color patterns, watermarks, holograms, and micro-printing to differentiate between real and fake banknotes with high accuracy.

## Project Overview

Counterfeit currency creates financial risks and affects economic stability. Traditional verification methods rely on human inspection, which is often slow and inaccurate.

This project automates currency verification by:

- Applying image preprocessing and feature extraction
- Using Machine Learning / CNN-based classification models
- ORB-based feature matching (optional)
- Training on a custom dataset of genuine and counterfeit notes

Suitable for deployment in banks, retail counters, ATMs, and currency sorting machines.

## Key Features

- Classifies currency as **Real** or **Fake**
- Supports multiple Indian denominations
- Works for both front and back currency images
- High accuracy using VGG16 transfer learning
- Custom dataset with lighting and angle variations

## Tech Stack

| Category        | Tools / Libraries |
|----------------|------------------|
| Language       | Python |
| Computer Vision | OpenCV, Pillow |
| Deep Learning  | TensorFlow / Keras / PyTorch |
| Classical ML   | Scikit-learn |
| Data Handling  | NumPy, Pandas |
| Development    | VS Code / Jupyter Notebook |
| Deployment (Optional) | Flask / Android Studio |

## Project Structure

Fake-Currency-Detection-System
│
├─ training-data/
├─ test-images/
├─ models/        → trained .h5 / .tflite models
│
├─ utils.py
├─ detection.py
├─ training_cnn.py
├─ README.md
└─ requirements.txt

## How It Works

1. User uploads/captures an image
2. Preprocessing: noise removal, grayscale, contrast enhancement
3. Feature extraction from edges, textures, watermark areas
4. CNN model performs classification
5. Output shows authenticity and confidence score

## Installation

```bash
git clone https://github.com/yourusername/Fake-Currency-Detection-System.git
cd Fake-Currency-Detection-System
pip install -r requirements.txt

Usage

python detection.py --image sample.jpg

Example Output:

Prediction: REAL (98.4% confidence)
Currency: ₹200 (Front Side)

Dataset Details
	•	Custom collected dataset with real & fake Indian currency notes
	•	Includes ₹50, ₹100, ₹200, ₹500 denominations
	•	Images captured at various angles and lighting conditions
	•	Balanced classes to avoid bias
	•	Resized to 224×224 for model training
	•	Labeled using directory format or CSV

Counterfeit samples are used strictly for research and educational purposes.

Results

Model Used	Accuracy	Notes
ORB Feature Matching	Good	Works only with exact known templates
CNN (VGG16 Transfer Learning)	90%+	Best performance and scalable
TensorFlow Lite Model	High	Optimized for mobile devices

Future Enhancements
	•	Add more global currencies
	•	Mobile app for real-time camera detection
	•	Support UV / multispectral imaging for hidden features
	•	Integrate Vision Transformer (ViT)
	•	Expand dataset for stronger generalization

References
	•	OpenCV Documentation
	•	TensorFlow / Keras Documentation
	•	RBI currency security guidelines
	•	Research articles on CNN-based counterfeit detection
