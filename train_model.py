import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

print("✅ Loading data...")

# Load features and labels
X = np.load("X.npy")
y = np.load("y.npy")

print(f"X shape: {X.shape}, y shape: {y.shape}")

# Build a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(13, 200, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("🧠 Training the model...")
model.fit(X, y, epochs=10, batch_size=1, verbose=1)

# Save model
model.save("deepfake_audio_detector.h5")
print("✅ Model saved as deepfake_audio_detector.h5")
