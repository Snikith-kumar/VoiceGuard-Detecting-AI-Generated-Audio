# VoiceGuard-Detecting-AI-Generated-Audio

This project, VoiceGuard, aims to address this growing threat by developing a robust detection framework that can distinguish between real and deepfake audio. Leveraging a hybrid deep learning model combining Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) units, the system is designed to capture both spatial and temporal patterns in audio data. Key features such as Mel Frequency Cepstral Coefficients (MFCCs) and additional spectral metrics provide the foundation for accurate classification. The ASVspoof 2019 dataset, which includes both genuine and manipulated audio samples, serves as the primary benchmark for training and evaluation.

![image](https://github.com/user-attachments/assets/a9f574c4-5729-4309-aa60-c1c8d44cc567)

Implementation of the project

Home Page and System Features Overview

![image](https://github.com/user-attachments/assets/daee12d9-f1d2-4d5c-abe7-f91edb3cb20a)

Fig indicates homepage of VoiceGuard highlights the system's purpose and key features. It outlines components like advanced detection, visual MFCC analysis, quick results, and its professional utility for verifying audio authenticity.

Upload Interface of VoiceGuard

![image](https://github.com/user-attachments/assets/6c91cbab-3220-4b28-be35-1c79aba9bfe1)

Fig illustrates this screen allows users to upload audio files in WAV, FLAC, or MP3 format for analysis. The drag-and-drop interface supports files up to 200MB and directs the uploaded file to the detection pipeline.

Results

![image](https://github.com/user-attachments/assets/504c09d7-99d2-4500-9a86-60df0164109f)


![image](https://github.com/user-attachments/assets/f9117d27-b935-4ff6-a13e-71c00227e00f)


Steps to Run:

1.Clone the Repo 2.Install all the required libaries and Packages

pip install -r requirements.txt

3. Run the Application
   
streamlit run audio.py







