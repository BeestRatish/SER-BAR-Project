import os
import glob
import numpy as np
import librosa
import time
import psutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, LSTM, Bidirectional, 
                                    Dense, Dropout, Input, Flatten, 
                                    Concatenate, Multiply, Add, LayerNormalization,
                                    GlobalAveragePooling1D, Attention)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns

# ====================== CONFIGURATION ======================
DATA_DIR = "data/cremad"  # Directory containing CREMA-D WAV files
MODEL_PATH = "crema_hybrid_emotion_model.h5"
BEST_MODEL = "best_hybrid_crema_model.h5"

# CREMA-D emotion mapping (6 classes)
EMOTION_MAP = {
    'ANG': 'angry',
    'DIS': 'disgust',
    'FEA': 'fear',
    'HAP': 'happy',
    'NEU': 'neutral',
    'SAD': 'sad'
}

# Audio processing parameters
SAMPLE_RATE = 22050  
DURATION = 3  
N_MFCC = 64
N_CHROMA = 24
N_MEL = 128          
N_CONTRAST = 6
N_TONNETZ = 6

# ====================== DATA PROCESSING ======================
def validate_filename(filename):
    """Check if filename matches CREMA-D format"""
    parts = filename.split('_')
    return len(parts) >= 3 and parts[2] in EMOTION_MAP

def extract_features(audio, sr):
    """Extract comprehensive audio features with error handling"""
    features = []
    
    try:
        # MFCCs with delta and delta-delta
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        features.extend(np.mean(mfcc.T, axis=0))
        features.extend(np.mean(delta_mfcc.T, axis=0))
        features.extend(np.mean(delta2_mfcc.T, axis=0))
        
        # Chroma with variants
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_cens = librosa.feature.chroma_cens(y=audio, sr=sr)
        chroma_cqt = librosa.feature.chroma_cqt(y=audio, sr=sr)
        features.extend(np.mean(chroma.T, axis=0))
        features.extend(np.mean(chroma_cens.T, axis=0))
        features.extend(np.mean(chroma_cqt.T, axis=0))
        
        # Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MEL)
        features.extend(np.mean(librosa.power_to_db(mel).T, axis=0))
        
        # Spectral contrast with safe parameters
        contrast = librosa.feature.spectral_contrast(
            y=audio, 
            sr=sr, 
            n_bands=N_CONTRAST,
            fmin=200.0
        )
        features.extend(np.mean(contrast.T, axis=0))
        
        # Tonnetz
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        features.extend(np.mean(tonnetz.T, axis=0))
        
        # Zero crossing rate
        features.append(np.mean(librosa.feature.zero_crossing_rate(y=audio)))
        
        # RMS energy
        features.append(np.mean(librosa.feature.rms(y=audio)))
        
    except Exception as e:
        print(f"\nFeature extraction error: {str(e)}")
        return None
    
    return np.array(features)

def load_crema_dataset():
    """Load dataset with detailed progress tracking"""
    X, y = [], []
    counts = {'valid': 0, 'skipped': 0, 'error': 0}
    
    all_files = list(glob.glob(os.path.join(DATA_DIR, "*.wav")))
    total_files = len(all_files)
    
    if total_files == 0:
        raise ValueError(f"No WAV files found in {DATA_DIR}")
    
    print(f"\nProcessing {total_files} files...")
    start_time = time.time()
    mem_start = psutil.Process().memory_info().rss
    
    for i, filepath in enumerate(all_files):
        filename = os.path.basename(filepath)
        
        # Print progress every 100 files
        if i % 100 == 0:
            elapsed = time.time() - start_time
            print(f"\rProcessed {i}/{total_files} files ({i/total_files:.1%}) | "
                  f"Elapsed: {elapsed:.1f}s | "
                  f"Valid: {counts['valid']}", end="", flush=True)
        
        # Validate filename
        if not validate_filename(filename):
            counts['skipped'] += 1
            continue
            
        emotion_code = filename.split('_')[2]
        
        try:
            # Load and validate audio
            audio, sr = librosa.load(filepath, sr=SAMPLE_RATE)
            if len(audio) < SAMPLE_RATE * 0.5:
                counts['skipped'] += 1
                continue
                
            # Preprocess audio
            audio, _ = librosa.effects.trim(audio, top_db=20)
            audio = librosa.util.fix_length(audio, size=SAMPLE_RATE * DURATION)
            
            # Extract features
            features = extract_features(audio, sr)
            if features is None:
                counts['error'] += 1
                continue
                
            X.append(features)
            y.append(EMOTION_MAP[emotion_code])
            counts['valid'] += 1
            
        except Exception as e:
            counts['error'] += 1
            continue
    
    # Final progress update
    elapsed = time.time() - start_time
    print(f"\rProcessed {total_files}/{total_files} files (100.0%) | "
          f"Elapsed: {elapsed:.1f}s | "
          f"Valid: {counts['valid']}", flush=True)
    
    print(f"\nDataset loaded: {counts['valid']} valid, {counts['skipped']} skipped, {counts['error']} errors")
    
    if counts['valid'] == 0:
        raise ValueError("No valid files processed")
    
    return np.array(X), np.array(y)

# ====================== MODEL ARCHITECTURE ======================
def create_hybrid_model(input_shape, num_classes):
    """Build CNN-LSTM model with proper shape handling"""
    inputs = Input(shape=input_shape)
    
    # CNN Branch
    x = Conv1D(256, 5, activation='relu', padding='same')(inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    
    x1 = Conv1D(128, 5, activation='relu', padding='same')(x)
    x1 = LayerNormalization()(x1)
    x1 = Dropout(0.3)(x1)
    x1 = MaxPooling1D(4)(x1)  # Shape: (None, 92, 128)
    
    # LSTM Branch - adjust to match CNN output shape
    x2 = Conv1D(256, 5, activation='relu', padding='same')(x)  # Additional conv to match dimensions
    x2 = MaxPooling1D(4)(x2)  # Shape: (None, 92, 256)
    x2 = Bidirectional(LSTM(128, return_sequences=True))(x2)  # Shape: (None, 92, 256)
    
    # Attention - adjust to match other branches
    x_att = Conv1D(128, 5, activation='relu', padding='same')(x1)
    x_att = LayerNormalization()(x_att)  # Shape: (None, 92, 128)
    
    # Feature fusion - now all branches have compatible shapes
    x_fused = Concatenate()([x1, x2, x_att])  # Now all inputs are (None, 92, *)
    x_fused = Conv1D(128, 3, activation='relu', padding='same')(x_fused)
    x_fused = LayerNormalization()(x_fused)
    
    # Output
    x_pool = GlobalAveragePooling1D()(x_fused)
    x_pool = Dense(128, activation='relu')(x_pool)
    x_pool = Dropout(0.4)(x_pool)
    outputs = Dense(num_classes, activation='softmax')(x_pool)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ====================== VISUALIZATION ======================
def plot_training(history):
    """Plot training metrics"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def plot_confusion(y_true, y_pred, classes):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.show()

# ====================== MAIN EXECUTION ======================
def main():
    print("Starting speech emotion recognition pipeline...")
    
    try:
        # Load data with progress tracking
        print("\n=== Loading Dataset ===")
        X, y = load_crema_dataset()
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        class_names = le.classes_
        print(f"\nClasses: {list(class_names)}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Reshape for CNN
        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)
        print(f"\nInput shape: {X_train.shape[1:]}")
        
        # Create model
        print("\n=== Building Model ===")
        model = create_hybrid_model(X_train.shape[1:], len(class_names))
        model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor='val_accuracy'),
            ModelCheckpoint(BEST_MODEL, save_best_only=True, monitor='val_accuracy'),
            ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6)
        ]
        
        # Train
        print("\n=== Training Model ===")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        print("\n=== Evaluation ===")
        model = tf.keras.models.load_model(BEST_MODEL)
        plot_training(history)
        
        y_pred = np.argmax(model.predict(X_test), axis=1)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        plot_confusion(y_test, y_pred, class_names)
        
        # Save model
        model.save(MODEL_PATH)
        print(f"\nModel saved to {MODEL_PATH}")
        
    except Exception as e:
        print(f"\nError in main pipeline: {str(e)}")
        print("\nDebugging Tips:")
        print(f"1. Verify {DATA_DIR} contains valid WAV files")
        print("2. Check CREMA-D filename format")
        print("3. Ensure dependencies are installed")
        print("4. Check available memory")

if __name__ == "__main__":
    main()