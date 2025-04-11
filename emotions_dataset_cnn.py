import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    SAMPLE_RATE = 22050
    DURATION = 3  # seconds
    SAMPLES = SAMPLE_RATE * DURATION
    DATA_PATH = "data/Emotions "
    SAVED_MODEL_PATH = "models/emotions_model.h5"
    BATCH_SIZE = 32
    EPOCHS = 50
    
# Feature extraction configuration
class FeatureConfig:
    N_MFCC = 40
    N_MELS = 128
    N_CHROMA = 12
    HOP_LENGTH = 512
    N_FFT = 2048

def load_and_preprocess_audio(file_path, sr=Config.SAMPLE_RATE, duration=Config.DURATION):
    """Load and preprocess audio file with fixed duration using soundfile backend."""
    try:
        # Load audio with specified duration using soundfile backend
        y, sr = librosa.load(file_path, sr=sr, duration=duration, res_type='kaiser_fast', backend='soundfile')
        
        # If audio is shorter than desired duration, pad with zeros
        if len(y) < Config.SAMPLES:
            y = np.pad(y, (0, Config.SAMPLES - len(y)), mode='constant')
        # If audio is longer, truncate
        else:
            y = y[:Config.SAMPLES]
            
        # Normalize audio
        y = librosa.util.normalize(y)
            
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        # Try alternative backend if soundfile fails
        try:
            y, sr = librosa.load(file_path, sr=sr, duration=duration, res_type='kaiser_fast', backend='audioread')
            if len(y) < Config.SAMPLES:
                y = np.pad(y, (0, Config.SAMPLES - len(y)), mode='constant')
            else:
                y = y[:Config.SAMPLES]
            y = librosa.util.normalize(y)
            return y, sr
        except Exception as e2:
            print(f"Both backends failed for {file_path}: {str(e2)}")
            return None, None

def extract_features(y, sr):
    """Extract multiple features from audio signal."""
    features = []
    
    if y is None:
        return None
    
    try:
        # 1. MFCC
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=FeatureConfig.N_MFCC,
            n_fft=FeatureConfig.N_FFT,
            hop_length=FeatureConfig.HOP_LENGTH
        )
        features.append(mfcc)

        # 2. Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr,
            n_mels=FeatureConfig.N_MELS,
            n_fft=FeatureConfig.N_FFT,
            hop_length=FeatureConfig.HOP_LENGTH
        )
        features.append(librosa.power_to_db(mel_spec))

        # 3. Chroma
        chroma = librosa.feature.chroma_stft(
            y=y, 
            sr=sr,
            n_chroma=FeatureConfig.N_CHROMA,
            n_fft=FeatureConfig.N_FFT,
            hop_length=FeatureConfig.HOP_LENGTH
        )
        features.append(chroma)

        # 4. Spectral Contrast
        contrast = librosa.feature.spectral_contrast(
            y=y, 
            sr=sr,
            n_fft=FeatureConfig.N_FFT,
            hop_length=FeatureConfig.HOP_LENGTH
        )
        features.append(contrast)

        # Stack all features
        features = np.vstack(features)
        return features
        
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return None

def augment_audio(y, sr):
    """Apply various augmentation techniques to audio."""
    augmented = []
    
    # Original audio
    augmented.append(y)
    
    # 1. Add noise
    noise_factor = 0.005
    noise = np.random.normal(0, noise_factor, len(y))
    augmented.append(y + noise)
    
    # 2. Time stretch
    stretch_rates = [0.8, 1.2]
    for rate in stretch_rates:
        augmented.append(librosa.effects.time_stretch(y, rate=rate))
    
    # 3. Pitch shift
    pitch_shifts = [-2, 2]  # semitones
    for n_steps in pitch_shifts:
        augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps))
    
    return augmented

def create_model(input_shape, num_classes):
    """Create CNN model with regularization and batch normalization."""
    model = models.Sequential([
        # First Conv Block
        layers.Conv2D(64, (3, 3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Second Conv Block
        layers.Conv2D(128, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Third Conv Block
        layers.Conv2D(256, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(512, kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def plot_training_history(history):
    """Plot training history with improved visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('emotions_training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix with improved visualization."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('emotions_confusion_matrix.png')
    plt.close()

def main():
    print("Starting Emotion Recognition Training Pipeline...")
    
    # 1. Load and preprocess data
    print("\nLoading and preprocessing data...")
    X = []
    y = []
    
    for emotion_folder in os.listdir(Config.DATA_PATH):
        emotion_path = os.path.join(Config.DATA_PATH, emotion_folder)
        if os.path.isdir(emotion_path):
            print(f"Processing {emotion_folder}...")
            for audio_file in os.listdir(emotion_path):
                if audio_file.endswith('.wav'):
                    file_path = os.path.join(emotion_path, audio_file)
                    
                    # Load and preprocess audio
                    audio, sr = load_and_preprocess_audio(file_path)
                    if audio is not None:
                        # Extract features
                        features = extract_features(audio, sr)
                        if features is not None:
                            # Apply augmentation
                            augmented_audio = augment_audio(audio, sr)
                            for aug_audio in augmented_audio:
                                aug_features = extract_features(aug_audio, sr)
                                if aug_features is not None:
                                    X.append(aug_features)
                                    y.append(emotion_folder)
    
    X = np.array(X)
    y = np.array(y)
    
    # 2. Prepare data for training
    print("\nPreparing data for training...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Reshape data for CNN input (add channel dimension)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    
    # 3. Create and compile model
    print("\nCreating and compiling model...")
    model = create_model(
        input_shape=(X_train.shape[1], X_train.shape[2], 1),
        num_classes=len(le.classes_)
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 4. Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'best_emotions_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001,
            verbose=1
        )
    ]
    
    # 5. Train the model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # 6. Evaluate and visualize results
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Plot training history
    plot_training_history(history)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred_classes, le.classes_)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=le.classes_))
    
    # Save the model
    model.save('emotions_model.h5')
    print("\nModel saved as 'emotions_model.h5'")

if __name__ == "__main__":
    main()