import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Conv1D, Activation, Dropout, Dense, Flatten, MaxPooling1D
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import regularizers
from matplotlib import pyplot as plt
import seaborn as sn
import pandas as pd
import sklearn.metrics as metrics
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Set data directory
data_directory = "data/ravdess"

# Define emotions dictionary
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry', 
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Feature extraction function
def extract_feature(data, sr, mfcc=True, chroma=True, mel=True):
    """
    Extract feature from audio data
    """
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

# Data augmentation functions
def noise(data, noise_factor):
    """
    Add random noise to sound
    """
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def shift(data, sr, shift_factor, direction='right'):
    """
    Shift the audio data
    """
    shift = int(sr * shift_factor)
    if direction == 'right':
        augmented_data = np.roll(data, shift)
    else:
        augmented_data = np.roll(data, -shift)
    return augmented_data

def stretch(data, rate=1.1):
    """
    Stretch the audio data
    """
    return librosa.effects.time_stretch(data, rate=rate)

def pitch(data, sr, pitch_factor=0.7):
    """
    Pitch shift the audio data
    """
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=pitch_factor)

# Load data function with augmentation
def load_data(save=False, augment=True):
    """
    Loading dataset with optional data augmentation
    """
    x, y = [], []
    for file in glob.glob(data_directory + "/Actor_*/*.wav"):
        # Load audio file
        data, sr = librosa.load(file)
        
        # Extract features
        feature = extract_feature(data, sr, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        
        # Get emotion label from filename
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        y.append(emotion)
        
        if augment:
            # Add noise
            n_data = noise(data, 0.001)
            n_feature = extract_feature(n_data, sr, mfcc=True, chroma=True, mel=True)
            x.append(n_feature)
            y.append(emotion)
            
            # Shift the data
            s_data = shift(data, sr, 0.25, 'right')
            s_feature = extract_feature(s_data, sr, mfcc=True, chroma=True, mel=True)
            x.append(s_feature)
            y.append(emotion)
            
            # Add pitch variation (new augmentation)
            p_data = pitch(data, sr, 0.7)
            p_feature = extract_feature(p_data, sr, mfcc=True, chroma=True, mel=True)
            x.append(p_feature)
            y.append(emotion)
    
    if save:
        np.save('X', np.array(x))
        np.save('y', np.array(y))
        
    return np.array(x), np.array(y)

# Create model function
def create_model(input_shape):
    """
    Create the CNN model with the same architecture as the original
    """
    model = Sequential()
    model.add(Conv1D(256, 5, padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5, padding='same', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 5, padding='same', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5, padding='same', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(units=8,
                    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                    bias_regularizer=regularizers.l2(1e-4),
                    activity_regularizer=regularizers.l2(1e-5)
                   ))
    model.add(Activation('softmax'))
    
    # Use Adam optimizer with learning rate scheduler
    opt = keras.optimizers.Adam(learning_rate=0.001)
    
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model

# Plot training history function
def plot_training_history(history):
    """
    Plot the training and validation accuracy and loss
    """
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# Plot confusion matrix function
def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot confusion matrix with better visualization
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize
    
    plt.figure(figsize=(12, 10))
    
    # Plot raw counts
    plt.subplot(1, 2, 1)
    sn.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Counts)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Plot normalized percentages
    plt.subplot(1, 2, 2)
    sn.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Normalized)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

# Main function
def main():
    print("Loading and preprocessing data...")
    
    # Try to load saved data, if not available, process from raw files
    try:
        X = np.load("X.npy")
        y = np.load("y.npy", allow_pickle=True)
        print("Loaded preprocessed data from files.")
    except:
        print("Processing raw audio files...")
        X, y = load_data(save=True, augment=True)
    
    # Encode the labels
    print("Encoding labels...")
    labelencoder = LabelEncoder()
    y_encoded = labelencoder.fit_transform(y)
    
    # Get class names for later use
    class_names = list(labelencoder.classes_)
    print(f"Classes: {class_names}")
    
    # Split data into training and testing sets
    print("Splitting data into train and test sets...")
    x_train, x_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded)
    
    # Print dataset information
    print(f"Training set size: {x_train.shape}")
    print(f"Testing set size: {x_test.shape}")
    print(f"Number of features: {x_train.shape[1]}")
    
    # Reshape data for CNN input
    X_train_cnn = np.expand_dims(x_train, axis=2)
    X_test_cnn = np.expand_dims(x_test, axis=2)
    
    # Create model
    print("Creating model...")
    model = create_model((x_train.shape[1], 1))
    model.summary()
    
    # Define callbacks for training
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1)
    
    # Train model with more epochs and improved batch size
    print("Training model...")
    history = model.fit(
        X_train_cnn, y_train,
        batch_size=128,  # Improved batch size (original was 64)
        epochs=50,      # More epochs (original was 100)
        validation_data=(X_test_cnn, y_test),
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=1
    )
    
    # Load the best model
    model = keras.models.load_model('best_model.h5')
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model on test set
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(X_test_cnn, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Get predictions
    y_pred_prob = model.predict(X_test_cnn)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Calculate detailed metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    print(f"\nDetailed Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro): {recall_macro:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, class_names)
    
    # Perform cross-validation
    print("\nPerforming 5-fold cross-validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold+1}/5")
        
        # Split data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y_encoded[train_idx], y_encoded[val_idx]
        
        # Reshape for CNN
        X_train_fold = np.expand_dims(X_train_fold, axis=2)
        X_val_fold = np.expand_dims(X_val_fold, axis=2)
        
        # Create and train model
        fold_model = create_model((X.shape[1], 1))
        fold_model.fit(
            X_train_fold, y_train_fold,
            batch_size=128,
            epochs=100,  # Reduced epochs for cross-validation
            verbose=0
        )
        
        # Evaluate
        _, accuracy = fold_model.evaluate(X_val_fold, y_val_fold, verbose=0)
        cv_scores.append(accuracy)
        print(f"Fold {fold+1} accuracy: {accuracy:.4f}")
    
    print(f"\nCross-validation results:")
    print(f"CV scores: {cv_scores}")
    print(f"Mean CV accuracy: {np.mean(cv_scores):.4f}")
    print(f"Standard deviation: {np.std(cv_scores):.4f}")
    
    # Save the model
    model.save('speech_emotion_recognition_model.h5')
    print("\nModel saved as 'speech_emotion_recognition_model.h5'")

# Run the main function
if __name__ == "__main__":
    main()
