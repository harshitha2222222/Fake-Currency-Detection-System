# Fake Currency Detection System

A Computer Vision-based system that detects counterfeit currency using OpenCV and machine learning/deep learning models. It identifies security features such as texture, color patterns, watermarks, holograms, and micro-printing to differentiate between real and fake banknotes with high accuracy.

## Project Overview
Counterfeit currency creates financial risks and affects economic stability. Manual inspection is slow and inaccurate.

This project automates currency verification by:
- Image preprocessing and feature extraction
- Machine Learning / CNN-based classification models
- ORB-based feature matching (optional)
- Training on a custom dataset of genuine and counterfeit notes

Suitable for deployment in banks, retail counters, ATMs, and currency sorting machines.

## Key Features
- Detects Real vs Fake Indian currency
- Supports multiple denominations
- Front and back side recognition
- High accuracy using VGG16 Transfer Learning
- Custom dataset with different lighting/angles

## Tech Stack
| Category | Tools / Libraries |
|---------|------------------|
| Programming Language | Python |
| Computer Vision | OpenCV, Pillow |
| Deep Learning | TensorFlow / Keras / PyTorch |
| Machine Learning | Scikit-learn |
| Data Handling | NumPy, Pandas |
| Development | VS Code, Jupyter Notebook |
| Deployment (Optional) | Flask / Android Studio |

## Project Structure
Fake-Currency-Detection-System  
│  
├── training-data/  
├── test-images/  
├── models/                (trained .h5 / .tflite models)  
│  
├── utils.py  
├── detection.py  
├── training_cnn.py  
├── README.md  
└── requirements.txt  

## How It Works
1. Capture or upload currency image  
2. Preprocessing: grayscale, noise removal, enhancement  
3. Feature extraction (texture, edges, watermark area)  
4. CNN classification  
5. Output: Real or Fake with confidence score  

## Installation
Clone repo and install dependencies:
git clone https://github.com/yourusername/Fake-Currency-Detection-System.git  
cd Fake-Currency-Detection-System  
pip install -r requirements.txt  

## Usage
Run prediction script:
python detection.py --image sample.jpg  

Example Output:
Prediction: REAL (98.4% confidence)  
Currency: ₹200 (Front side)  

## Dataset Details
- Custom dataset of Indian currency notes (real + fake)
- Denominations used: ₹50, ₹100, ₹200, ₹500
- Captured under various lighting conditions and angles
- Balanced dataset to avoid bias
- Images resized to 224x224 for model training
- Labeled via directories / CSV

Note: Counterfeit samples are only for research and educational purposes.

## Results
| Model | Accuracy | Notes |
|------|----------|------|
| ORB Feature Matching | Good | Requires stable template |
| CNN (VGG16 Transfer Learning) | 90%+ | Best recognition performance |
| TensorFlow Lite Mobile Model | High | Optimized for phones |

## Future Enhancements
- Support for global currencies
- Real-time mobile scanning app
- UV / IR imaging for hidden security features
- Vision Transformer (ViT) integration
- Larger dataset for higher robustness

## References
- OpenCV Documentation
- TensorFlow / Keras Documentation
- RBI Security Feature Guidelines
- Research papers on image-based currency authentication
