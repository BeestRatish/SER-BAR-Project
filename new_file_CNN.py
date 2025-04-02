import librosa
import soundfile
import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import sklearn.metrics as metrics

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, regularizers
from tensorflow.keras.layers import (Conv1D, Activation, Dropout, 
                                   Dense, Flatten, MaxPooling1D)
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, 
                                      ReduceLROnPlateau)
from tensorflow.keras.optimizers import Adam

# ====================== CONSTANTS AND CONFIGURATION ======================
DATA_DIR = "data/ravdess"
MODEL_SAVE_PATH = "speech_emotion_recognition_model.h5"
BEST_MODEL_PATH = "best_model.h5"

EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# ====================== DATA PROCESSING FUNCTIONS ======================
def extract_feature(data, sr, mfcc=True, chroma=True, mel=True):
    """Extract features from audio data"""
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

def load_data(save=False, augment=True):
    """Load dataset with optional augmentation"""
    x, y = [], []
    
    for file in glob.glob(os.path.join(DATA_DIR, "Actor_*/*.wav")):
        data, sr = librosa.load(file)
        feature = extract_feature(data, sr, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        
        file_name = os.path.basename(file)
        emotion = EMOTIONS[file_name.split("-")[2]]
        y.append(emotion)
        
        if augment:
            # Noise augmentation
            n_data = noise(data, 0.001)
            n_feature = extract_feature(n_data, sr, mfcc=True, chroma=True, mel=True)
            x.append(n_feature)
            y.append(emotion)
            
            # Time shift augmentation
            s_data = shift(data, sr, 0.25, 'right')
            s_feature = extract_feature(s_data, sr, mfcc=True, chroma=True, mel=True)
            x.append(s_feature)
            y.append(emotion)
            
            # Pitch shift augmentation
            p_data = pitch(data, sr, 0.7)
            p_feature = extract_feature(p_data, sr, mfcc=True, chroma=True, mel=True)
            x.append(p_feature)
            y.append(emotion)
    
    if save:
        np.save('X', np.array(x))
        np.save('y', np.array(y))
        
    return np.array(x), np.array(y)

# ====================== DATA AUGMENTATION FUNCTIONS ======================
def noise(data, noise_factor):
    """Add random noise to audio"""
    noise = np.random.randn(len(data))
    return (data + noise_factor * noise).astype(type(data[0]))

def shift(data, sr, shift_factor, direction='right'):
    """Time shift audio"""
    shift = int(sr * shift_factor)
    return np.roll(data, shift) if direction == 'right' else np.roll(data, -shift)

def stretch(data, rate=1.1):
    """Time stretch audio"""
    return librosa.effects.time_stretch(data, rate=rate)

def pitch(data, sr, pitch_factor=0.7):
    """Pitch shift audio"""
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=pitch_factor)

# ====================== MODEL FUNCTIONS ======================
def create_model(input_shape):
    """Create and compile the CNN model"""
    model = Sequential([
        Conv1D(256, 5, padding='same', input_shape=input_shape),
        Activation('relu'),
        Conv1D(128, 5, padding='same', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)),
        Activation('relu'),
        Dropout(0.1),
        MaxPooling1D(pool_size=8),
        Conv1D(128, 5, padding='same', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)),
        Activation('relu'),
        Conv1D(128, 5, padding='same', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)),
        Activation('relu'),
        Dropout(0.5),
        Flatten(),
        Dense(units=8,
              kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
              bias_regularizer=regularizers.l2(1e-4),
              activity_regularizer=regularizers.l2(1e-5)),
        Activation('softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def get_callbacks():
    """Return list of training callbacks"""
    return [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        ModelCheckpoint(BEST_MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1)
    ]

# ====================== VISUALIZATION FUNCTIONS ======================
def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
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
    plt.savefig('training_history.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10))
    
    # Counts plot
    sn.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    # Normalized plot
    sn.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

# ====================== EVALUATION FUNCTIONS ======================
def evaluate_model(model, X_test, y_test, class_names):
    """Evaluate model performance"""
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'f1': f1_score(y_test, y_pred, average='macro')
    }
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    plot_confusion_matrix(y_test, y_pred, class_names)
    
    return metrics

def cross_validate(X, y, input_shape, n_splits=5):
    """Perform k-fold cross validation"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold+1}/{n_splits}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Reshape for CNN
        X_train = np.expand_dims(X_train, axis=2)
        X_val = np.expand_dims(X_val, axis=2)
        
        model = create_model(input_shape)
        model.fit(X_train, y_train, batch_size=128, epochs=50, verbose=0)
        
        _, accuracy = model.evaluate(X_val, y_val, verbose=0)
        cv_scores.append(accuracy)
        print(f"Fold {fold+1} accuracy: {accuracy:.4f}")
    
    print(f"\nCV results - Mean: {np.mean(cv_scores):.4f}, Std: {np.std(cv_scores):.4f}")
    return cv_scores

# ====================== MAIN FUNCTION ======================
def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    try:
        X = np.load("X.npy")
        y = np.load("y.npy", allow_pickle=True)
        print("Loaded preprocessed data from files.")
    except:
        print("Processing raw audio files...")
        X, y = load_data(save=True, augment=True)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = list(label_encoder.classes_)
    print(f"Classes: {class_names}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
    )
    
    # Reshape for CNN
    X_train_cnn = np.expand_dims(X_train, axis=2)
    X_test_cnn = np.expand_dims(X_test, axis=2)
    
    # Create and train model
    model = create_model((X_train.shape[1], 1))
    model.summary()
    
    history = model.fit(
        X_train_cnn, y_train,
        batch_size=128,
        epochs=400,
        validation_data=(X_test_cnn, y_test),
        callbacks=get_callbacks(),
        verbose=1
    )
    
    # Load best model and evaluate
    model = keras.models.load_model(BEST_MODEL_PATH)
    plot_training_history(history)
    evaluate_model(model, X_test_cnn, y_test, class_names)
    
    # Cross-validation
    cv_scores = cross_validate(X, y_encoded, (X_train.shape[1], 1))
    
    # Save final model
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()