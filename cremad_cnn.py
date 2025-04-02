import os
import glob
import numpy as np
import librosa
from sklearn.model_selection import train_test_split, KFold
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

# Audio processing parameters (updated for better feature extraction)
SAMPLE_RATE = 22050  
DURATION = 3  
N_MFCC = 64          # Increased from 40
N_CHROMA = 24        # Increased from 12
N_MEL = 128          
N_CONTRAST = 7       # New spectral contrast feature
N_TONNETZ = 6        # New tonnetz feature

# ====================== DATA PROCESSING ======================
def validate_filename(filename):
    """Check if filename matches CREMA-D format"""
    parts = filename.split('_')
    return len(parts) >= 3 and parts[2] in EMOTION_MAP

def extract_features(audio, sr):
    """Extract comprehensive audio features"""
    features = []
    
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
    
    # Spectral contrast with adjusted parameters
    try:
        contrast = librosa.feature.spectral_contrast(
            y=audio, 
            sr=sr, 
            n_bands=6,  # Reduced from 7
            fmin=200.0  # Increased minimum frequency
        )
        features.extend(np.mean(contrast.T, axis=0))
    except Exception as e:
        print(f"Spectral contrast error: {str(e)}")
        features.extend([0] * 6)  # Pad with zeros if extraction fails
    
    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
    features.extend(np.mean(tonnetz.T, axis=0))
    
    # Zero crossing rate
    features.append(np.mean(librosa.feature.zero_crossing_rate(y=audio)))
    
    # RMS energy
    features.append(np.mean(librosa.feature.rms(y=audio)))
    
    return np.array(features)

def load_crema_dataset():
    """Load and validate CREMA-D dataset"""
    X, y = [], []
    valid_files = 0
    skipped_files = 0
    
    for filepath in glob.glob(os.path.join(DATA_DIR, "*.wav")):
        filename = os.path.basename(filepath)
        
        # Validate filename structure
        if not validate_filename(filename):
            print(f"Skipping invalid filename: {filename}")
            skipped_files += 1
            continue
            
        emotion_code = filename.split('_')[2]
        
        try:
            # Load and validate audio
            audio, sr = librosa.load(filepath, sr=SAMPLE_RATE)
            if len(audio) < SAMPLE_RATE * 0.5:  # At least 0.5s of audio
                print(f"Skipping short audio file: {filename}")
                skipped_files += 1
                continue
                
            # Preprocess audio
            audio, _ = librosa.effects.trim(audio, top_db=20)
            audio = librosa.util.fix_length(audio, size=SAMPLE_RATE * DURATION)
            
            # Extract features
            features = extract_features(audio, sr)
            X.append(features)
            y.append(EMOTION_MAP[emotion_code])
            valid_files += 1
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            skipped_files += 1
            continue
    
    print(f"\nDataset loaded: {valid_files} valid files, {skipped_files} skipped")
    
    if valid_files == 0:
        raise ValueError("No valid files found. Please check:\n"
                       f"1. Files exist in {DATA_DIR}\n"
                       "2. Filenames match CREMA-D format (e.g., 1001_DFA_ANG_XX.wav)\n"
                       "3. Audio files are not corrupted")
    
    return np.array(X), np.array(y)

# ====================== VISUALIZATION ======================
def plot_training(history):
    """Plot training history"""
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

# ====================== HYBRID MODEL ARCHITECTURE ======================
def attention_block(inputs, time_steps):
    """Attention mechanism for temporal weighting"""
    # Query, Key, Value projections
    query = Dense(time_steps, activation='softmax')(inputs)
    key = Dense(time_steps)(inputs)
    value = Dense(time_steps)(inputs)
    
    # Scaled dot-product attention
    attention = Multiply()([query, key])
    attention = tf.keras.layers.Softmax(axis=-1)(attention)
    weighted = Multiply()([attention, value])
    
    return weighted

def create_hybrid_model(input_shape, num_classes):
    """Build hybrid CNN-LSTM model with attention"""
    inputs = Input(shape=input_shape)
    
    # CNN Branch
    x = Conv1D(256, 5, activation='relu', padding='same')(inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    
    x1 = Conv1D(128, 5, activation='relu', padding='same')(x)
    x1 = LayerNormalization()(x1)
    x1 = Dropout(0.3)(x1)
    x1 = MaxPooling1D(4)(x1)
    
    # LSTM Branch
    x2 = Bidirectional(LSTM(128, return_sequences=True))(x)
    x2 = LayerNormalization()(x2)
    x2 = Dropout(0.3)(x2)
    
    # Attention mechanism
    time_steps = x1.shape[1]
    x_att = attention_block(x1, time_steps)
    
    # Feature fusion
    x_fused = Concatenate()([x1, x2, x_att])
    x_fused = Conv1D(128, 3, activation='relu', padding='same')(x_fused)
    x_fused = LayerNormalization()(x_fused)
    
    # Skip connection
    x_res = Conv1D(128, 1)(x)  # Projection for skip connection
    x_res = MaxPooling1D(4)(x_res)
    x_out = Add()([x_fused, x_res])
    
    # Global pooling and dense layers
    x_pool = GlobalAveragePooling1D()(x_out)
    x_pool = Dense(128, activation='relu')(x_pool)
    x_pool = LayerNormalization()(x_pool)
    x_pool = Dropout(0.4)(x_pool)
    
    outputs = Dense(num_classes, activation='softmax')(x_pool)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ====================== MAIN EXECUTION ======================
def main():
    try:
        # Load and validate data
        print("Loading CREMA-D dataset...")
        X, y = load_crema_dataset()
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        class_names = le.classes_
        print(f"\nClasses found: {list(class_names)}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Reshape for CNN (add channel dimension)
        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)
        print(f"\nInput shape: {X_train.shape[1:]}")
        
        # Create and train hybrid model
        model = create_hybrid_model(X_train.shape[1:], len(class_names))
        model.summary()
        
        # Enhanced callbacks
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True, monitor='val_accuracy'),
            ModelCheckpoint(BEST_MODEL, save_best_only=True, monitor='val_accuracy'),
            ReduceLROnPlateau(factor=0.2, patience=8, min_lr=1e-6),
            tf.keras.callbacks.TensorBoard(log_dir='./logs')
        ]
        
        # Train with class weights (handle imbalanced data)
        class_counts = np.bincount(y_train)
        class_weights = {i: 1./count for i, count in enumerate(class_counts)}
        
        print("\nTraining hybrid model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=150,  # Increased for deeper model
            batch_size=64,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Load best model and evaluate
        model = tf.keras.models.load_model(BEST_MODEL)
        plot_training(history)
        
        y_pred = np.argmax(model.predict(X_test), axis=1)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        plot_confusion(y_test, y_pred, class_names)
        
        # Save final model
        model.save(MODEL_PATH)
        print(f"\nModel saved to {MODEL_PATH}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nDebugging Tips:")
        print(f"1. Verify {DATA_DIR} contains .wav files")
        print("2. Check filenames match pattern: [ID]_[Actor]_[EMO]_[Intensity].wav")
        print("3. Ensure audio files are playable")
        print("4. Check librosa version (pip install librosa==0.9.2)")

if __name__ == "__main__":
    main()