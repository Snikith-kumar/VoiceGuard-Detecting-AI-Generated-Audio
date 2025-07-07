import os
import numpy as np
import librosa

print("✅ Script is running...")

# Lists to store features and labels
X = []
y = []

# Fixed parameters
fixed_length = 200       # max number of time frames for MFCC
n_mfcc = 13              # number of MFCC features
sr = 16000               # sample rate for all audio

# Loop through fake and real folders
for label, folder in enumerate(['fake', 'real']):
    folder_path = os.path.join("dataset", folder)
    print(f"📂 Reading folder: {folder_path}")
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            print(f"🎧 Processing: {file_path}")
            audio, _ = librosa.load(file_path, sr=sr)

            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
            if mfcc.shape[1] < fixed_length:
                pad_width = fixed_length - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
            else:
                mfcc = mfcc[:, :fixed_length]

            X.append(mfcc)
            y.append(label)

# Convert lists to NumPy arrays
X = np.array(X)[..., np.newaxis]  # Add channel dimension
y = np.array(y)

# Save to .npy files
np.save("X.npy", X)
np.save("y.npy", y)

print("✅ Feature extraction complete!")
print(f"✅ X shape: {X.shape}, y shape: {y.shape}")
