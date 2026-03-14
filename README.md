# Handwritten Digit Recognition

A CNN-based digit recognizer trained on MNIST, served via a FastAPI backend and containerized with Docker. Includes an interactive frontend where you draw a digit and get a real-time prediction.

![Demo](frontend/demo.png)

## Features

- 99.2% validation accuracy on MNIST
- 4-layer CNN with Batch Normalization and Dropout
- Real-world preprocessing pipeline (adaptive thresholding, morphological ops, ROI extraction)
- REST API built with FastAPI
- Fully containerized with Docker
- Interactive draw-and-predict frontend (no framework, pure HTML/JS)

## Tech Stack

Python, TensorFlow/Keras, OpenCV, FastAPI, Docker, NumPy

## Project Structure
```
├── digit_cnn_enhanced.h5   # Trained model
├── server.py               # FastAPI server
├── requirements.txt
├── Dockerfile
├── hdr.ipynb               # Training notebook
└── frontend/
    └── index.html          # Draw and predict UI
```

## Run Locally

**Prerequisites:** Docker installed
```bash
# Clone the repo
git clone https://github.com/abhay0132/Handwritten-Digit-Recognition.git
cd Handwritten-Digit-Recognition

# Build and run the container
docker build -t digit-recognizer .
docker run -p 8000:8000 digit-recognizer
```

Then open `frontend/index.html` in your browser, draw a digit, and hit **Predict**.

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Upload an image, returns predicted digit and confidence |

**Example response:**
```json
{
  "digit": 3,
  "confidence": 100.0,
  "all_probabilities": [...]
}
```

## Model

Trained on 60,000 MNIST images with data augmentation (rotation, zoom, translation, shear). Input images are preprocessed to match MNIST format before inference.
