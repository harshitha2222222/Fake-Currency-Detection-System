Fake Currency Detection System

A Computer Vision-based system that detects counterfeit currency using OpenCV and machine learning/deep learning models. It analyzes currency security features such as texture, color patterns, watermarks, holograms, and micro–printing to distinguish between real and fake banknotes with high accuracy.

Project Overview

Counterfeit currency creates financial risks and affects economic stability. Traditional verification methods rely on human expertise, which can be slow and inaccurate.

This system automates currency verification by:
	•	Applying image preprocessing and feature extraction
	•	Using Machine Learning or CNN-based classification models
	•	Integrating ORB-based feature matching (optional)
	•	Training on a custom dataset of genuine and counterfeit notes

The model can be deployed in banks, billing counters, and ATM/currency counting machines.

Key Features
	•	Classifies currency as Real or Fake
	•	Supports multiple Indian denominations
	•	Works on both front and back note images
	•	High accuracy with VGG16-based deep learning model
	•	Custom dataset captured in different environmental conditions

Tech Stack

Category	Tools / Libraries
Language	Python
Computer Vision	OpenCV, Pillow
Deep Learning	TensorFlow / Keras / PyTorch
Classical ML	Scikit-learn
Data Handling	NumPy, Pandas
Development	VS Code / Jupyter Notebook
Deployment (Optional)	Flask, Android Studio

Project Structure

Fake-Currency-Detection-System
│
├─ training-data/
├─ test-images/
├─ models/                → trained .h5 / .tflite models
│
├─ utils.py
├─ detection.py
├─ training_cnn.py
├─ README.md
└─ requirements.txt

(Folder naming can be adjusted as needed.)

How It Works
	1.	Currency image captured or uploaded
	2.	Preprocessing: noise removal, cropping, grayscale conversion
	3.	Feature extraction using edges, textures, watermark areas
	4.	Machine learning model predicts authenticity
	5.	Output: “Real” or “Fake” with confidence score



Usage

python detection.py --image sample.jpg

Example output:

Prediction: REAL (98.4% confidence)
Currency: ₹200 Front

Dataset Details
	•	Custom dataset with Indian real and fake currency images
	•	Includes ₹50, ₹100, ₹200, ₹500 denominations
	•	Collected at various lighting and angle conditions
	•	Balanced number of real vs. fake samples
	•	Resized to 224×224 resolution for CNN training
	•	Dataset labeling stored in folders or CSV format

Counterfeit samples used purely for research and academic learning.

Results

Model Used	Accuracy	Observation
ORB Feature Matching	Good	Works only on consistent known templates
CNN (VGG16 Transfer Learning)	90%+	Best overall performance
TensorFlow Lite Model	High	Suitable for mobile deployment

Future Enhancements
	•	Expand to multiple global currencies
	•	Deploy as a mobile app with real-time detection
	•	Support UV/multispectral imaging for hidden security features
	•	Use Vision Transformers (ViT) for higher accuracy
	•	Improve dataset size and generalization



References
	•	OpenCV Documentation
	•	TensorFlow / Keras Documentation
	•	Research works on counterfeit detection
	•	RBI currency design publications



