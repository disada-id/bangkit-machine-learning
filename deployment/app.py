from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import librosa
import os

app = Flask(__name__)

# Path to the trained model
model_path = 'model\checkpoint.h5'

# Mapping from class index to labels
label_mapping = {
    0: 'bayi sedang kesakitan',
    1: 'bayi sedang merasa kembung',
    2: 'bayi merasa kurang nyaman',
    3: 'bayi sedang lapar',
    4: 'bayi sedang lelah'
}

# Load the model
loaded_model = load_model(model_path)
model_input_shape = loaded_model.input_shape[1:]

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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded file from the mobile app
        audio_file = request.files['audio']

        # Save the file to a temporary location
        temp_file_path = 'temp_audio.wav'
        audio_file.save(temp_file_path)

        # Extract features from the uploaded audio
        audio_features = extract_features_new_audio(temp_file_path)

        # Perform prediction
        predicted_label, prediction_probabilities = predict_label(audio_features)

        # Prepare the results to be sent back to the mobile app
        results = {
            'predicted_label': predicted_label,
            'prediction_probabilities': {label: prob.item() for label, prob in zip(label_mapping.values(), prediction_probabilities)}
        }

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)