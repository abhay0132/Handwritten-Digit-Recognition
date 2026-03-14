import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
from PIL import Image

app = FastAPI(title="Handwritten Digit Recognition API")

# Allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
model = tf.keras.models.load_model("digit_cnn_enhanced.h5")


def preprocess(image_bytes: bytes) -> np.ndarray:
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Resize to manageable size if too large
    img = cv2.resize(img, (280, 280))

    # Adaptive thresholding
    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # Morphological operations to clean noise
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # Find bounding box (ROI extraction)
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]

    # Add padding and resize to 28x28 (centered)
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
    img = cv2.resize(img, (28, 28))

    # Normalize and reshape for model
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img


@app.get("/")
def root():
    return {"message": "Digit Recognition API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    processed = preprocess(image_bytes)
    predictions = model.predict(processed)
    digit = int(np.argmax(predictions))
    confidence = float(np.max(predictions)) * 100
    return JSONResponse({
        "digit": digit,
        "confidence": round(confidence, 2),
        "all_probabilities": predictions[0].tolist()
    })


@app.get("/health")
def health():
    return {"status": "ok"}
