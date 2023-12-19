from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import librosa
import os
import requests

app = Flask(__name__)

# Path to the trained model
model_path = os.path.join('model', 'checkpoint.h5')

# Mapping from class index to labels
label_mapping = {
    0: 'merasa kesakitan',
    1: 'sedang merasa kembung',
    2: 'merasa kurang nyaman',
    3: 'sedang lapar',
    4: 'sedang lelah'
}

# Load the model
loaded_model = load_model(model_path)
model_input_shape = loaded_model.input_shape[1:]

# Menentukan direktori untuk menyimpan file sementara
temp_audio_dir = 'disada_audio'
os.makedirs(temp_audio_dir, exist_ok=True)

def extract_features_new_audio(file_path):
    y, sr = librosa.load(file_path)

    # Extract audio features as before
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0])
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    zero_crossings = np.mean(librosa.feature.zero_crossing_rate(y))
    rmse = np.mean(librosa.feature.rms(y=y))
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

    # Concatenate features into a single vector
    feature_vector = np.concatenate([mfccs, [spectral_centroid, spectral_bandwidth],
                                    spectral_contrast, chroma, [zero_crossings, rmse, tempo]])

    # Reshape feature vector to match model input shape
    feature_vector = feature_vector.reshape(1, -1)

    return feature_vector

def predict_label(audio_features):
    # Continue with prediction as before
    prediction_probabilities = loaded_model.predict(audio_features)
    predicted_class = np.argmax(prediction_probabilities)
    predicted_label = label_mapping[predicted_class]

    return predicted_label, prediction_probabilities[0]

def get_food_recommendation(predicted_label):
    rekomendasi_panganan = {
        "merasa kesakitan": {
            "rekomendasi": ["ASI (Air Susu Ibu)", "Pisang", "Bubur ayam", "Semangka", "Yogurt"],
            "penjelasan": "ASI adalah pilihan utama untuk memberikan nutrisi dan kenyamanan. Pisang, bubur ayam, semangka, dan yogurt juga dapat membantu memberikan nutrisi yang mudah dicerna dan menenangkan."
        },
        "sedang merasa kembung": {
            "rekomendasi": ["ASI", "Pepaya", "Bubur labu kuning", "Apel", "Teh chamomile (dalam kadar rendah)"],
            "penjelasan": "ASI tetap menjadi pilihan utama. Pepaya, bubur labu kuning, apel, dan teh chamomile dalam kadar rendah dapat membantu mengurangi rasa kembung dan memberikan kenyamanan."
        },
        "merasa kurang nyaman": {
            "rekomendasi": ["ASI", "Avokad", "Bubur beras merah", "Pear", "Susu formula hypoallergenic"],
            "penjelasan": "ASI tetap menjadi pilihan utama. Avokad, bubur beras merah, pear, dan susu formula hypoallergenic dapat membantu memberikan nutrisi yang mudah dicerna dan cocok untuk bayi yang merasa kurang nyaman."
        },
        "sedang lapar": {
            "rekomendasi": ["ASI", "Susu formula", "Bubur susu", "Alpukat", "Telur rebus"],
            "penjelasan": "ASI atau susu formula sesuai dengan kebutuhan. Bubur susu, alpukat, dan telur rebus adalah sumber nutrisi yang baik untuk memenuhi kebutuhan nutrisi saat bayi sedang lapar."
        },
        "sedang lelah": {
            "rekomendasi": ["ASI", "Madu (dalam kadar rendah)", "Bubur quinoa", "Strawberry", "Smoothie buah"],
            "penjelasan": "ASI tetap menjadi pilihan utama. Madu dalam kadar rendah, bubur quinoa, strawberry, dan smoothie buah dapat memberikan energi tambahan dan nutrisi untuk bayi yang sedang lelah."
        }
    }

    return rekomendasi_panganan.get(predicted_label, {})

@app.route('/')
def index():
    return 'Aplikasi Mobile Sedang Berjalan'

@app.route('/predict', methods=['POST'])
def predict_mobile():
    try:
        # Check if the 'file' key is in the request
        if 'file' not in request.files:
            return jsonify({'error': 'File not found in request'})

        # Get the uploaded file from the request
        uploaded_file = request.files['file']

        # Check if the file has an allowed extension (adjust as needed)
        allowed_extensions = {'wav'}
        if '.' not in uploaded_file.filename or uploaded_file.filename.split('.')[-1].lower() not in allowed_extensions:
            return jsonify({'error': 'Invalid file format'})

        # Save the file to a temporary location
        temp_file_path = os.path.join(temp_audio_dir, 'temp_audio.wav')
        uploaded_file.save(temp_file_path)

        # Extract features from the uploaded audio
        audio_features = extract_features_new_audio(temp_file_path)

        # Perform prediction
        predicted_label, prediction_probabilities = predict_label(audio_features)

        # Get food recommendation
        food_recommendation = get_food_recommendation(predicted_label)

        # Display the results as JSON
        results = {
            'Hasil': predicted_label,
            'kemungkinan': {label: float(prob) for label, prob in zip(label_mapping.values(), prediction_probabilities)},
            'rekomendasi_panganan': food_recommendation
        }

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)