import os
import librosa
import numpy as np
import pandas as pd

# Fungsi untuk mengekstrak fitur audio
def extract_features(file_path):
    # Menggunakan librosa untuk mendapatkan fitur-fitur audio
    y, sr = librosa.load(file_path)

    # Mel-frequency cepstral coefficients (MFCCs)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)

    # Spektrum daya
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0])
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)

    # Chroma feature
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)

    # Zero-crossing rate
    zero_crossings = np.mean(librosa.feature.zero_crossing_rate(y))

    # RMS (Root Mean Square) energy
    rmse = np.mean(librosa.feature.rms(y=y))

    # Tempo
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

    # Menggabungkan semua fitur menjadi satu vektor
    feature_vector = np.concatenate([mfccs, [spectral_centroid, spectral_bandwidth],
                                    spectral_contrast, chroma, [zero_crossings, rmse, tempo]])

    return feature_vector

# Fungsi untuk menyimpan fitur dalam format CSV
def save_features_to_csv(folders, output_csv):
    data = []

    for folder in folders:
        label = os.path.basename(folder)
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            features = extract_features(file_path)
            data.append([label] + features.tolist())

    columns = ["Label"] + [f"Feature_{i}" for i in range(len(data[0]) - 1)]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)

# List folder yang berisi file audio
folders = [
    "/content/drive/MyDrive/Dataset/donateacry_corpus_cleaned_and_updated_data/belly_pain",
    "/content/drive/MyDrive/Dataset/donateacry_corpus_cleaned_and_updated_data/burping",
    "/content/drive/MyDrive/Dataset/donateacry_corpus_cleaned_and_updated_data/discomfort",
    "/content/drive/MyDrive/Dataset/donateacry_corpus_cleaned_and_updated_data/hungry",
    "/content/drive/MyDrive/Dataset/donateacry_corpus_cleaned_and_updated_data/tired",
]

# Path untuk menyimpan hasil ekstraksi fitur dalam format CSV
output_csv = "audio_features.csv"

# Menyimpan fitur dalam format CSV
save_features_to_csv(folders, output_csv)
