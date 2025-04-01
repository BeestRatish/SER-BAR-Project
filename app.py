import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
# Set matplotlib to use a non-interactive backend before importing plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = tf.keras.models.load_model('speech_emotion_recognition_model.h5')

# Define emotions dictionary (same as in training)
emotions = {
    0: 'angry',
    1: 'calm',
    2: 'disgust',
    3: 'fearful',
    4: 'happy',
    5: 'neutral',
    6: 'sad',
    7: 'surprised'
}

# Feature extraction function (same as in training)
def extract_feature(data, sr, mfcc=True, chroma=True, mel=True):
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(y=data, sr=sr).T, axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
        result = np.hstack((result, mel))
    return result

# Function to predict emotion from audio file
def predict_emotion(file_path):
    # Load audio file
    data, sr = librosa.load(file_path, sr=None)
    
    # Extract features
    features = extract_feature(data, sr)
    
    # Reshape for model input
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    features = np.expand_dims(features, axis=2)  # Add channel dimension
    
    # Predict
    predictions = model.predict(features)[0]
    
    # Get predicted emotion and confidence
    predicted_class = np.argmax(predictions)
    predicted_emotion = emotions[predicted_class]
    confidence = predictions[predicted_class] * 100
    
    # Get all emotion probabilities for visualization
    emotion_probs = {emotions[i]: float(predictions[i] * 100) for i in range(len(emotions))}
    
    return predicted_emotion, confidence, emotion_probs

# Function to create visualization
def create_emotion_plot(emotion_probs):
    # Sort emotions by probability
    sorted_emotions = dict(sorted(emotion_probs.items(), key=lambda item: item[1], reverse=True))
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    plt.style.use('dark_background')
    
    # Create bars with custom colors
    colors = ['#FF5252', '#4CAF50', '#9C27B0', '#2196F3', '#FFEB3B', '#607D8B', '#3F51B5', '#FF9800']
    bars = plt.bar(sorted_emotions.keys(), sorted_emotions.values(), color=colors[:len(sorted_emotions)])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.title('Emotion Prediction Confidence', fontsize=16, pad=20)
    plt.ylabel('Confidence (%)', fontsize=14)
    plt.xlabel('Emotion', fontsize=14)
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save plot to a base64 string to embed in HTML
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    return base64.b64encode(image_png).decode('utf-8')

# Function to create waveform visualization
def create_waveform_plot(file_path):
    # Load audio file
    data, sr = librosa.load(file_path, sr=None)
    
    # Create waveform plot
    plt.figure(figsize=(10, 3))
    plt.style.use('dark_background')
    
    # Plot waveform
    plt.plot(np.linspace(0, len(data)/sr, len(data)), data, color='#2196F3')
    
    plt.title('Audio Waveform', fontsize=14, pad=20)
    plt.ylabel('Amplitude', fontsize=12)
    plt.xlabel('Time (s)', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save plot to a base64 string to embed in HTML
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    return base64.b64encode(image_png).decode('utf-8')

# Function to create spectrogram visualization
def create_spectrogram_plot(file_path):
    # Load audio file
    data, sr = librosa.load(file_path, sr=None)
    
    # Create spectrogram plot
    plt.figure(figsize=(10, 4))
    plt.style.use('dark_background')
    
    # Compute spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
    
    # Plot spectrogram
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    
    plt.title('Spectrogram', fontsize=14, pad=20)
    plt.tight_layout()
    
    # Save plot to a base64 string to embed in HTML
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    return base64.b64encode(image_png).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Predict emotion
            emotion, confidence, emotion_probs = predict_emotion(file_path)
            
            # Create visualizations
            emotion_plot = create_emotion_plot(emotion_probs)
            waveform_plot = create_waveform_plot(file_path)
            
            # Try to create spectrogram (might fail if librosa.display is not available)
            try:
                spectrogram_plot = create_spectrogram_plot(file_path)
            except:
                spectrogram_plot = None
            
            return jsonify({
                'success': True,
                'emotion': emotion,
                'confidence': float(confidence),
                'emotion_probs': emotion_probs,
                'emotion_plot': emotion_plot,
                'waveform_plot': waveform_plot,
                'spectrogram_plot': spectrogram_plot,
                'audio_path': f'/static/uploads/{filename}'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
