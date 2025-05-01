🧠 Handwritten Digit Recognition using CNN (MNIST)
This project implements a robust Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset, achieving ~99.2% accuracy on the validation set. It also includes a preprocessing pipeline using OpenCV that handles noisy or imperfect digit images for real-world robustness.

🔧 Features
✅ 4-layer CNN with Batch Normalization and Dropout for regularization

🧹 Image preprocessing: adaptive thresholding, morphological operations, dynamic ROI extraction, and 28x28 centering

🔄 Data augmentation: rotation, zoom, translation, and shear using ImageDataGenerator

🧪 Trained and evaluated on 60,000+ MNIST images

🧠 Saved trained model as .h5 and ready for prediction via image upload

📦 Tech Stack
Python, TensorFlow/Keras, OpenCV, NumPy, Matplotlib

CNN architecture with ReLU, Softmax, Adam optimizer

Model visualization and prediction support
