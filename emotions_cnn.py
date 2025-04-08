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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Add this function after the import statements but before the data directory declaration
def validate_data_directory(dir_path):
    """
    Validate that the data directory exists and contains the expected emotion folders
    """
    if not os.path.exists(dir_path):
        print(f"Error: Data directory '{dir_path}' does not exist.")
        return False
    
    if not os.path.isdir(dir_path):
        print(f"Error: '{dir_path}' is not a directory.")
        return False
    
    # Try to list the contents of the directory
    try:
        contents = os.listdir(dir_path)
        emotion_folders = [item for item in contents if os.path.isdir(os.path.join(dir_path, item)) and not item.startswith('.')]
        
        if len(emotion_folders) == 0:
            print(f"Error: No emotion folders found in '{dir_path}'.")
            return False
        
        print(f"Found the following emotion folders: {emotion_folders}")
        return True
    except Exception as e:
        print(f"Error accessing directory '{dir_path}': {str(e)}")
        return False

# Set data directory - check both with and without trailing space
data_directory = os.path.join("data", "Emotions")
if not os.path.exists(data_directory):
    alt_directory = os.path.join("data", "Emotions ")
    if os.path.exists(alt_directory):
        data_directory = alt_directory
        print(f"Using directory with trailing space: {data_directory}")
    else:
        print(f"Warning: Neither '{data_directory}' nor '{alt_directory}' exists. Will attempt to find the correct path during validation.")

# Define emotions dictionary - TO BE UPDATED based on your actual emotions
emotions = {
    'Angry': 'angry',
    'Disgusted': 'disgusted',
    'Fearful': 'fearful',
    'Happy': 'happy',
    'Neutral': 'neutral',
    'Sad': 'sad',
    'Suprised': 'surprised'
}

def extract_feature(data, sr, mfcc=True, chroma=True, mel=True):
    """
    Extract feature from audio data with enhanced parameters
    """
    result = np.array([])
    if mfcc:
        # Increase n_mfcc for more detailed coefficients and add deltas
        mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
        mfccs_delta = librosa.feature.delta(mfccs)
        mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
        
        # Use mean and standard deviation for better feature representation
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        mfccs_delta_mean = np.mean(mfccs_delta.T, axis=0)
        mfccs_delta_std = np.std(mfccs_delta.T, axis=0)
        mfccs_delta2_mean = np.mean(mfccs_delta2.T, axis=0)
        mfccs_delta2_std = np.std(mfccs_delta2.T, axis=0)
        
        result = np.hstack((result, mfccs_mean, mfccs_std, mfccs_delta_mean, mfccs_delta_std, mfccs_delta2_mean, mfccs_delta2_std))
    
    if chroma:
        # Improved chroma parameters
        chroma = librosa.feature.chroma_stft(y=data, sr=sr, n_chroma=24)
        chroma_mean = np.mean(chroma.T, axis=0)
        chroma_std = np.std(chroma.T, axis=0)
        result = np.hstack((result, chroma_mean, chroma_std))
    
    if mel:
        # Add more mel bands and power-to-db conversion for better representation
        mel_spec = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_mean = np.mean(mel_db.T, axis=0)
        mel_std = np.std(mel_db.T, axis=0)
        result = np.hstack((result, mel_mean, mel_std))
    
    # Add spectral features for better emotion discrimination
    spec_cent = librosa.feature.spectral_centroid(y=data, sr=sr)
    spec_cent_mean = np.mean(spec_cent.T, axis=0)
    spec_cent_std = np.std(spec_cent.T, axis=0)
    
    spec_bw = librosa.feature.spectral_bandwidth(y=data, sr=sr)
    spec_bw_mean = np.mean(spec_bw.T, axis=0)
    spec_bw_std = np.std(spec_bw.T, axis=0)
    
    rolloff = librosa.feature.spectral_rolloff(y=data, sr=sr)
    rolloff_mean = np.mean(rolloff.T, axis=0)
    rolloff_std = np.std(rolloff.T, axis=0)
    
    zcr = librosa.feature.zero_crossing_rate(data)
    zcr_mean = np.mean(zcr.T, axis=0)
    zcr_std = np.std(zcr.T, axis=0)
    
    # Add all spectral features
    result = np.hstack((result, 
                        spec_cent_mean, spec_cent_std,
                        spec_bw_mean, spec_bw_std,
                        rolloff_mean, rolloff_std,
                        zcr_mean, zcr_std))
    
    return result

# Data augmentation functions
def noise(data, noise_factor):
    """
    Add random noise to sound
    """
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
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

def load_data(save=False, augment=True, max_files_per_emotion=None):
    """
    Loading dataset with improved augmentation for better quality training data
    
    Parameters:
    -----------
    save : bool
        Whether to save the processed data to disk
    augment : bool
        Whether to apply data augmentation
    max_files_per_emotion : int or None
        If set, limits the number of files processed per emotion category
    """
    x, y = [], []
    emotion_counts = {}  # Track file counts per emotion
    
    try:
        # Handle directory with potential space in name
        print(f"Loading data from {data_directory}")
        
        # Process each emotion folder
        emotion_folders = []
        for item in os.listdir(data_directory):
            folder_path = os.path.join(data_directory, item)
            if os.path.isdir(folder_path) and not item.startswith('.'):
                emotion_folders.append((folder_path, item))
        
        if not emotion_folders:
            print("No emotion folders found!")
            return np.array([]), np.array([])
        
        print(f"Found {len(emotion_folders)} emotion folders")
        
        # Balance dataset across emotions if needed
        if max_files_per_emotion is None:
            # Find minimum files count to balance dataset
            min_files = float('inf')
            for emotion_dir, emotion_name in emotion_folders:
                wav_files = glob.glob(os.path.join(emotion_dir, "*.wav"))
                count = len(wav_files)
                if count > 0 and count < min_files:
                    min_files = count
            
            # Adjust max_files to ensure balanced dataset (but not too small)
            max_files_per_emotion = max(min_files, 50)  # At least 50 files per emotion if available
        
        print(f"Using up to {max_files_per_emotion} files per emotion for balanced dataset")
        
        # Process each emotion folder
        for emotion_dir, emotion_name in emotion_folders:
            print(f"Processing {emotion_name} files...")
            wav_files = glob.glob(os.path.join(emotion_dir, "*.wav"))
            total_files = len(wav_files)
            emotion_counts[emotion_name] = total_files
            
            if not wav_files:
                print(f"No WAV files found for {emotion_name}")
                continue
                
            print(f"Found {total_files} WAV files in {emotion_name}")
            
            # Limit files if needed for balancing
            selected_files = wav_files
            if max_files_per_emotion and total_files > max_files_per_emotion:
                print(f"Limiting to {max_files_per_emotion} files for balanced dataset")
                # Use stratified sampling when possible
                import random
                random.seed(42)  # For reproducibility
                selected_files = random.sample(wav_files, max_files_per_emotion)
            
            for file in selected_files:
                # Load audio file
                try:
                    # Load with consistent sample rate
                    data, sr = librosa.load(file, sr=22050)
                    
                    # Normalize audio
                    data = librosa.util.normalize(data)
                    
                    # Ensure minimum length (pad if needed)
                    min_samples = sr * 2  # At least 2 seconds
                    if len(data) < min_samples:
                        data = librosa.util.fix_length(data, size=min_samples)
                    
                    # Add original audio features
                    feature = extract_feature(data, sr, mfcc=True, chroma=True, mel=True)
                    x.append(feature)
                    y.append(emotion_name)
                    
                    if augment:
                        # Advanced augmentation techniques
                        
                        # 1. Time stretching (slower and faster)
                        for stretch_rate in [0.9, 1.1]:
                            stretched_data = librosa.effects.time_stretch(data, rate=stretch_rate)
                            # Ensure correct length after stretching
                            if len(stretched_data) < sr * 1.5:  # At least 1.5 seconds
                                continue
                            stretched_feature = extract_feature(stretched_data, sr, mfcc=True, chroma=True, mel=True)
                            x.append(stretched_feature)
                            y.append(emotion_name)
                        
                        # 2. Pitch shifting (up and down)
                        for pitch_steps in [-2, 2]:  # Semitones
                            pitched_data = librosa.effects.pitch_shift(data, sr=sr, n_steps=pitch_steps)
                            pitched_feature = extract_feature(pitched_data, sr, mfcc=True, chroma=True, mel=True)
                            x.append(pitched_feature)
                            y.append(emotion_name)
                        
                        # 3. Adding noise at different levels
                        for noise_level in [0.005, 0.01]:
                            noisy_data = data + noise_level * np.random.randn(len(data))
                            noisy_data = noisy_data.astype(type(data[0]))
                            noisy_feature = extract_feature(noisy_data, sr, mfcc=True, chroma=True, mel=True)
                            x.append(noisy_feature)
                            y.append(emotion_name)
                        
                        # 4. Time shifting
                        for shift_percent in [0.1, 0.2]:
                            shift_samples = int(len(data) * shift_percent)
                            shifted_data = np.roll(data, shift_samples)
                            shifted_feature = extract_feature(shifted_data, sr, mfcc=True, chroma=True, mel=True)
                            x.append(shifted_feature)
                            y.append(emotion_name)
                        
                        # 5. Combined augmentations (more realistic)
                        # Pitch shift + noise
                        pitched_noisy_data = librosa.effects.pitch_shift(data, sr=sr, n_steps=1.5)
                        pitched_noisy_data = pitched_noisy_data + 0.005 * np.random.randn(len(pitched_noisy_data))
                        pitched_noisy_feature = extract_feature(pitched_noisy_data, sr, mfcc=True, chroma=True, mel=True)
                        x.append(pitched_noisy_feature)
                        y.append(emotion_name)
                        
                        # Time stretch + shift
                        stretched_shifted_data = librosa.effects.time_stretch(data, rate=0.95)
                        if len(stretched_shifted_data) > sr * 1.5:  # Ensure enough length
                            stretched_shifted_data = np.roll(stretched_shifted_data, int(sr * 0.1))
                            stretched_shifted_feature = extract_feature(stretched_shifted_data, sr, mfcc=True, chroma=True, mel=True)
                            x.append(stretched_shifted_feature)
                            y.append(emotion_name)
                    
                except Exception as e:
                    print(f"Error processing file {file}: {str(e)}")
                    continue
        
        # Print dataset statistics
        print("\nEmotion distribution in raw dataset:")
        for emotion, count in emotion_counts.items():
            print(f"{emotion}: {count} audio files")
        
        augmented_counts = {}
        for emotion_label in y:
            augmented_counts[emotion_label] = augmented_counts.get(emotion_label, 0) + 1
        
        print("\nEmotion distribution after augmentation:")
        for emotion, count in augmented_counts.items():
            print(f"{emotion}: {count} samples")
            
    except Exception as e:
        print(f"Error accessing directory {data_directory}: {str(e)}")
        print("Please ensure that the data directory path is correct, including any spaces in the folder name.")
        return np.array([]), np.array([])
    
    if len(x) == 0:
        print("No data was processed. Please check the data directory path and file structure.")
        return np.array([]), np.array([])
    
    if save:
        np.save('X_emotions', np.array(x))
        np.save('y_emotions', np.array(y))
        print(f"Saved preprocessed data: {len(x)} samples")
        
    return np.array(x), np.array(y)

def squeeze_excitation_block(input_tensor, ratio=8):
    """
    Squeeze and Excitation block for channel-wise attention
    """
    channels = input_tensor.shape[-1]
    
    # Squeeze operation (global average pooling)
    x = tf.keras.layers.GlobalAveragePooling1D()(input_tensor)
    
    # Excitation operation (two FC layers)
    x = Dense(channels // ratio, activation='relu')(x)
    x = Dense(channels, activation='sigmoid')(x)
    
    # Reshape and multiply
    x = tf.keras.layers.Reshape((1, channels))(x)
    x = tf.keras.layers.multiply([input_tensor, x])
    
    return x

def residual_block(x, filters, kernel_size=3, stride=1, use_se=True):
    """
    Improved residual block with optional squeeze-excitation
    """
    shortcut = x
    
    # First convolution
    x = Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # Second convolution
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Squeeze and Excitation block
    if use_se:
        x = squeeze_excitation_block(x)
    
    # Handle different dimensions in shortcut
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    
    # Add shortcut
    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    
    return x

def create_model(input_shape):
    """
    Create the CNN model with the same architecture as improved_cnn.py
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

def plot_training_history(history):
    """
    Plot enhanced training history with more metrics and better styling
    """
    # Set style for better visualization
    plt.style.use('ggplot')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, color='#0072B2')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='#D55E00', linestyle='--')
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].legend(loc='lower right', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss', linewidth=2, color='#0072B2')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#D55E00', linestyle='--')
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].legend(loc='upper right', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot learning rate if available
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'], label='Learning Rate', linewidth=2, color='#009E73')
        axes[1, 0].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot validation metrics
    val_acc = history.history['val_accuracy']
    
    # Calculate trend line
    epochs = np.arange(len(val_acc))
    z = np.polyfit(epochs, val_acc, 1)
    p = np.poly1d(z)
    
    axes[1, 1].plot(val_acc, label='Validation Accuracy', linewidth=2, color='#D55E00')
    axes[1, 1].plot(epochs, p(epochs), "--", label=f'Trend (slope: {z[0]:.5f})', color='#CC79A7')
    axes[1, 1].set_title('Validation Accuracy Trend', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Accuracy', fontsize=12)
    axes[1, 1].legend(loc='lower right', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('emotions_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\nTraining Summary:")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f} (epoch {np.argmax(history.history['val_accuracy'])+1})")
    print(f"Best training accuracy: {max(history.history['accuracy']):.4f} (epoch {np.argmax(history.history['accuracy'])+1})")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    
    # Calculate stability metrics
    val_acc_last_10 = history.history['val_accuracy'][-10:]
    val_acc_stability = np.std(val_acc_last_10)
    print(f"Validation accuracy stability (std dev of last 10 epochs): {val_acc_stability:.4f}")

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot enhanced confusion matrix with better visualization and metrics
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize
    
    # Set style for better visualization
    plt.style.use('ggplot')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
                yticklabels=class_names, ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Predicted Label', fontsize=12)
    axes[0, 0].set_ylabel('True Label', fontsize=12)
    
    # Plot normalized confusion matrix (percentages)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, 
                yticklabels=class_names, ax=axes[0, 1])
    axes[0, 1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Predicted Label', fontsize=12)
    axes[0, 1].set_ylabel('True Label', fontsize=12)
    
    # Plot per-class metrics
    # Calculate precision, recall for each class
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    # Replace NaN values with 0
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1 = np.nan_to_num(f1)
    
    # Create a DataFrame for easier plotting
    metrics_df = pd.DataFrame({
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }, index=class_names)
    
    # Plot the metrics
    metrics_df.plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Emotion', fontsize=12)
    axes[1, 0].set_ylabel('Score', fontsize=12)
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].legend(loc='lower right')
    
    # Plot error distribution
    error_indices = y_true != y_pred
    errors_by_class = {}
    
    for true, pred in zip(y_true[error_indices], y_pred[error_indices]):
        true_class = class_names[true]
        pred_class = class_names[pred]
        error_key = f"{true_class} → {pred_class}"
        errors_by_class[error_key] = errors_by_class.get(error_key, 0) + 1
    
    # Sort errors by frequency
    errors_sorted = sorted(errors_by_class.items(), key=lambda x: x[1], reverse=True)
    top_errors = dict(errors_sorted[:10])  # Get top 10 error types
    
    # Plot top errors
    axes[1, 1].bar(top_errors.keys(), top_errors.values(), color='#D55E00')
    axes[1, 1].set_title('Top 10 Confusion Pairs', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('True → Predicted', fontsize=12)
    axes[1, 1].set_ylabel('Count', fontsize=12)
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('emotions_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary of confusion statistics
    print("\nConfusion Matrix Analysis:")
    print(f"Overall accuracy: {np.sum(np.diag(cm)) / np.sum(cm):.4f}")
    
    # Print per-class statistics
    for i, class_name in enumerate(class_names):
        print(f"\n{class_name}:")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall: {recall[i]:.4f}")
        print(f"  F1 Score: {f1[i]:.4f}")
        print(f"  Misclassifications: {np.sum(cm[i, :]) - cm[i, i]} out of {np.sum(cm[i, :])}")
    
    # Print most common error types
    print("\nTop 5 most common confusion pairs:")
    for i, (error_pair, count) in enumerate(errors_sorted[:5]):
        print(f"  {i+1}. {error_pair}: {count} instances")

def main():
    print("Loading and preprocessing data...")
    
    try:
        X = np.load("X_emotions.npy")
        y = np.load("y_emotions.npy", allow_pickle=True)
        print("Loaded preprocessed data from files.")
    except:
        print("Processing raw audio files...")
        X, y = load_data(save=True, augment=True)
    
    print(f"Dataset size: {len(X)} samples")
    
    # Encode labels
    labelencoder = LabelEncoder()
    y_encoded = labelencoder.fit_transform(y)
    class_names = list(labelencoder.classes_)
    num_classes = len(class_names)
    
    # Print class distribution
    class_counts = np.bincount(y_encoded)
    print("\nClass distribution:")
    for i, count in enumerate(class_counts):
        print(f"{class_names[i]}: {count} samples ({count/len(y_encoded)*100:.1f}%)")
    
    # Split data with stratification
    x_train, x_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=0.15,
        random_state=42,
        stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    # Reshape for CNN
    X_train_cnn = np.expand_dims(x_train_scaled, axis=2)
    X_test_cnn = np.expand_dims(x_test_scaled, axis=2)
    
    # Create model
    model = create_model((x_train.shape[1], 1))
    model.summary()
    
    # Enhanced callbacks
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        'best_emotions_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Class weights for imbalanced data
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    # Train with improved settings
    history = model.fit(
        X_train_cnn, y_train,
        validation_data=(X_test_cnn, y_test),
        epochs=20,
        batch_size=32,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Load best model
    model = keras.models.load_model('best_emotions_model.h5')
    
    # Evaluate and plot
    plot_training_history(history)
    
    # Get predictions and metrics
    y_pred = np.argmax(model.predict(X_test_cnn), axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, class_names)
    
    # Save the scaler for inference
    with open('emotions_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\nModel and scaler saved successfully!")
    return model, history

if __name__ == "__main__":
    main() 