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
from tqdm import tqdm
import gc
warnings.filterwarnings('ignore')

# Configuration
class Config:
    DATA_PATH = "data/Emotions "  # Directory name with a space
    SAVED_MODEL_PATH = "models/emotions_model_improved.h5"
    BATCH_SIZE = 32
    EPOCHS = 50
    NUM_WORKERS = 4    # Number of parallel workers
    
# Feature extraction configuration
class FeatureConfig:
    N_MFCC = 40
    N_MELS = 128
    N_CHROMA = 12
    HOP_LENGTH = 512
    N_FFT = 2048

def get_optimal_duration(file_paths):
    """Calculate optimal duration based on dataset statistics"""
    print("Calculating optimal duration from dataset...")
    durations = []
    for file_path in tqdm(file_paths):
        try:
            y, sr = librosa.load(file_path, duration=None)
            durations.append(len(y)/sr)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    optimal_duration = np.percentile(durations, 95)
    print(f"Optimal duration: {optimal_duration:.2f} seconds")
    return optimal_duration

def load_and_preprocess_audio(file_path, duration=None):
    """Load and preprocess audio file with adaptive duration."""
    try:
        # Load audio with full duration
        y, sr = librosa.load(file_path, sr=None, duration=duration)
        
        # Normalize audio
        y = librosa.util.normalize(y)
        
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None, None

def extract_features(y, sr):
    """Extract multiple features from audio signal."""
    if y is None:
        return None
    
    try:
        # Initialize feature list
        features = []
        
        # 1. MFCC with delta and delta-delta
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=FeatureConfig.N_MFCC,
            n_fft=FeatureConfig.N_FFT,
            hop_length=FeatureConfig.HOP_LENGTH
        )
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        # Ensure consistent length by padding or truncating
        target_length = 273  # Fixed length for all features
        
        # Function to pad or truncate features to target length
        def pad_or_truncate(feature, target_len):
            if feature.shape[1] < target_len:
                # Pad with zeros
                pad_width = ((0, 0), (0, target_len - feature.shape[1]))
                return np.pad(feature, pad_width, mode='constant')
            else:
                # Truncate
                return feature[:, :target_len]
        
        mfcc = pad_or_truncate(mfcc, target_length)
        delta_mfcc = pad_or_truncate(delta_mfcc, target_length)
        delta2_mfcc = pad_or_truncate(delta2_mfcc, target_length)
        
        features.extend([mfcc, delta_mfcc, delta2_mfcc])

        # 2. Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr,
            n_mels=FeatureConfig.N_MELS,
            n_fft=FeatureConfig.N_FFT,
            hop_length=FeatureConfig.HOP_LENGTH
        )
        mel_spec = librosa.power_to_db(mel_spec)
        mel_spec = pad_or_truncate(mel_spec, target_length)
        features.append(mel_spec)

        # 3. Chroma
        chroma = librosa.feature.chroma_stft(
            y=y, 
            sr=sr,
            n_chroma=FeatureConfig.N_CHROMA,
            n_fft=FeatureConfig.N_FFT,
            hop_length=FeatureConfig.HOP_LENGTH
        )
        chroma = pad_or_truncate(chroma, target_length)
        features.append(chroma)

        # 4. Spectral Contrast
        contrast = librosa.feature.spectral_contrast(
            y=y, 
            sr=sr,
            n_fft=FeatureConfig.N_FFT,
            hop_length=FeatureConfig.HOP_LENGTH
        )
        contrast = pad_or_truncate(contrast, target_length)
        features.append(contrast)

        # 5. Tonnetz features
        y_harmonic = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
        # Pad tonnetz features to match other features
        tonnetz = pad_or_truncate(tonnetz, target_length)
        features.append(tonnetz)

        # Stack all features and ensure consistent shape
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
    
    # 1. Add noise with different intensities
    noise_factors = [0.005, 0.01]
    for noise_factor in noise_factors:
        noise = np.random.normal(0, noise_factor, len(y))
        augmented.append(y + noise)
    
    # 2. Time stretch with more variations
    stretch_rates = [0.8, 0.9, 1.1, 1.2]
    for rate in stretch_rates:
        augmented.append(librosa.effects.time_stretch(y, rate=rate))
    
    # 3. Pitch shift with more variations
    pitch_shifts = [-2, -1, 1, 2]  # semitones
    for n_steps in pitch_shifts:
        augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps))
    
    # 4. Time shift
    shift_steps = [int(sr * 0.1), int(sr * 0.2)]  # 0.1 and 0.2 seconds
    for shift in shift_steps:
        augmented.append(np.roll(y, shift))
    
    return augmented

class AudioDataGenerator(tf.keras.utils.Sequence):
    """Custom data generator for audio processing that inherits from tf.keras.utils.Sequence"""
    def __init__(self, file_paths, labels, duration, batch_size=32, shuffle=True):
        self.file_paths = file_paths
        self.labels = labels
        self.duration = duration
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_paths = [self.file_paths[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]
        
        X_batch = []
        y_batch = []
        
        for file_path, label in zip(batch_paths, batch_labels):
            # Load and preprocess audio
            audio, sr = load_and_preprocess_audio(file_path, self.duration)
            if audio is None:
                continue
                
            # Extract features
            features = extract_features(audio, sr)
            if features is None:
                continue
                
            # Apply augmentation
            augmented_audio = augment_audio(audio, sr)
            for aug_audio in augmented_audio:
                aug_features = extract_features(aug_audio, sr)
                if aug_features is not None:
                    X_batch.append(aug_features)
                    y_batch.append(label)
        
        if not X_batch:  # If no valid features were extracted
            # Return a dummy batch with correct shape
            dummy_shape = (416, 273)  # Based on the error message
            return np.zeros((self.batch_size, *dummy_shape, 1)), np.zeros(self.batch_size)
        
        # Convert to numpy arrays with consistent shape
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        
        # Add channel dimension for CNN
        X_batch = X_batch[..., np.newaxis]
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

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
        
        # Fourth Conv Block
        layers.Conv2D(512, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.5),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(1024, kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
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
    plt.savefig('emotions_training_history_improved.png')
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
    plt.savefig('emotions_confusion_matrix_improved.png')
    plt.close()

def main():
    print("Starting Improved Emotion Recognition Training Pipeline...")
    
    # 1. Collect all file paths
    print("\nCollecting file paths...")
    all_files = []
    all_labels = []
    
    # Check if directory exists
    if not os.path.exists(Config.DATA_PATH):
        print(f"Error: Directory '{Config.DATA_PATH}' does not exist.")
        print("Available directories in 'data/':")
        for item in os.listdir("data"):
            print(f"  - {item}")
        return
    
    for emotion_folder in os.listdir(Config.DATA_PATH):
        emotion_path = os.path.join(Config.DATA_PATH, emotion_folder)
        if os.path.isdir(emotion_path):
            print(f"Found emotion folder: {emotion_folder}")
            for audio_file in os.listdir(emotion_path):
                if audio_file.endswith('.wav'):
                    all_files.append(os.path.join(emotion_path, audio_file))
                    all_labels.append(emotion_folder)
    
    print(f"Total files found: {len(all_files)}")
    
    if len(all_files) == 0:
        print("No audio files found. Please check the dataset structure.")
        return
    
    # 2. Calculate optimal duration
    optimal_duration = get_optimal_duration(all_files)
    
    # 3. Prepare data for training
    print("\nPreparing data for training...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(all_labels)
    
    # Split the data
    X_train_paths, X_test_paths, y_train, y_test = train_test_split(
        all_files, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Create data generators
    train_generator = AudioDataGenerator(
        X_train_paths, y_train, optimal_duration, 
        batch_size=Config.BATCH_SIZE, shuffle=True
    )
    
    test_generator = AudioDataGenerator(
        X_test_paths, y_test, optimal_duration, 
        batch_size=Config.BATCH_SIZE, shuffle=False
    )
    
    # Get a sample batch to determine input shape
    sample_batch_X, _ = train_generator[0]
    input_shape = (sample_batch_X.shape[1], sample_batch_X.shape[2], 1)
    
    # 4. Create and compile model
    print("\nCreating and compiling model...")
    model = create_model(
        input_shape=input_shape,
        num_classes=len(le.classes_)
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 5. Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'best_emotions_model_improved.h5',
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
    
    # 6. Train the model
    print("\nTraining model...")
    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=Config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # 7. Evaluate and visualize results
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Generate predictions
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Plot training history
    plot_training_history(history)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred_classes, le.classes_)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=le.classes_))
    
    # Save the model
    model.save('emotions_model_improved.h5')
    print("\nModel saved as 'emotions_model_improved.h5'")

if __name__ == "__main__":
    main() 