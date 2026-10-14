# Traffic Sign Recognition using CNN

## Overview
This project uses a Convolutional Neural Network (CNN) to recognize traffic signs from images. The model is trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset and predicts the traffic sign class for an uploaded image.

## Live Demo
🚀 [Try the Traffic Sign Recognition App](https://traffic-sign-prediction-yqc.streamlit.app/)

## Technologies Used
- Python
- TensorFlow and Keras
- CNN (Convolutional Neural Network)
- NumPy
- Streamlit
- Pillow

## Features
- Upload a traffic sign image.
- Predict the traffic sign class.
- Display prediction confidence.
- Show the top 3 predictions.

## Model Performance
- Test Accuracy: **98.07%**

## Project Files
- `Traffic_Sign_Recognition_CNN.ipynb` – Data preprocessing, model training, and evaluation.
- `app.py` – Streamlit web application.
- `traffic_sign_cnn.keras` – Trained CNN model.
- `class_names.json` – Traffic sign class labels.
- `requirements.txt` – Required Python libraries.

## How to Run Locally

1. Clone the repository.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   streamlit run app.py
   ```

## Dataset
German Traffic Sign Recognition Benchmark (GTSRB).

## Author
Developed as a Deep Learning project using CNNs.
