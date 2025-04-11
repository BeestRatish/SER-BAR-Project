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
    DATA_PATH = "data/Emotions "  # Keep the space as this is what works
    SAVED_MODEL_PATH = "models/emotions_model_improved.h5"
    BATCH_SIZE = 32
    EPOCHS = 50  # Keep 50 epochs
    NUM_WORKERS = 0  # Keep at 0 to avoid multiprocessing issues
    MAX_FILES_PER_CLASS = 200  # Increased files per class for better training
    SAMPLE_RATE = 22050  # Increased sample rate for better quality
    DURATION = 3.0  # Increased duration to capture more of each audio clip
    
# Feature extraction configuration
class FeatureConfig:
    N_MFCC = 40  # More MFCC features
    N_CHROMA = 24
    N_MELS = 128  # Increased for better spectral resolution
    N_CONTRAST = 7
    HOP_LENGTH = 512
    N_FFT = 2048  # Increased for better frequency resolution

def load_and_preprocess_audio(file_path, duration=None, sr=Config.SAMPLE_RATE):
    """Load and preprocess audio file with more robust preprocessing and multiple backend support."""
    try:
        # Try soundfile backend first
        y, sr = librosa.load(file_path, sr=sr, duration=duration, res_type='kaiser_fast', backend='soundfile')
        
        # Ensure consistent length
        target_length = int(sr * Config.DURATION)
        if len(y) < target_length:
            # Pad short samples
            y = np.pad(y, (0, target_length - len(y)), 'constant')
        else:
            # Truncate long samples
            y = y[:target_length]
        
        # Apply preprocessing techniques
        # 1. Normalize audio
        y = librosa.util.normalize(y)
        
        # 2. Apply pre-emphasis filter to emphasize higher frequencies
        y = librosa.effects.preemphasis(y)
        
        # 3. Trim silence - this helps focus on the important parts
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # 4. Ensure consistent length again after trimming
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), 'constant')
        else:
            y = y[:target_length]
            
        return y, sr
    except Exception as e:
        print(f"Soundfile backend failed for {file_path}: {str(e)}")
        try:
            # Try audioread backend as fallback
            y, sr = librosa.load(file_path, sr=sr, duration=duration, res_type='kaiser_fast', backend='audioread')
            
            target_length = int(sr * Config.DURATION)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), 'constant')
            else:
                y = y[:target_length]
            
            y = librosa.util.normalize(y)
            y = librosa.effects.preemphasis(y)
            y, _ = librosa.effects.trim(y, top_db=20)
            
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), 'constant')
            else:
                y = y[:target_length]
                
            return y, sr
        except Exception as e2:
            print(f"Both backends failed for {file_path}: {str(e2)}")
            return None, None

def extract_features(y, sr):
    """Extract comprehensive audio features with additional spectral features."""
    if y is None:
        return None
    
    try:
        # MFCCs with delta and delta-delta
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=FeatureConfig.N_MFCC,
            n_fft=FeatureConfig.N_FFT,
            hop_length=FeatureConfig.HOP_LENGTH
        )
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr,
            n_mels=FeatureConfig.N_MELS,
            n_fft=FeatureConfig.N_FFT,
            hop_length=FeatureConfig.HOP_LENGTH
        )
        mel_spec = librosa.power_to_db(mel_spec)
        
        # Spectral features - removing spectral contrast which causes Nyquist frequency error
        
        # 1. Spectral centroid - measure of spectral "brightness"
        centroid = librosa.feature.spectral_centroid(
            y=y, 
            sr=sr,
            n_fft=FeatureConfig.N_FFT,
            hop_length=FeatureConfig.HOP_LENGTH
        )
        
        # 2. Spectral rolloff - frequency below which most of energy is contained
        rolloff = librosa.feature.spectral_rolloff(
            y=y, 
            sr=sr,
            n_fft=FeatureConfig.N_FFT,
            hop_length=FeatureConfig.HOP_LENGTH
        )
        
        # 3. Spectral flatness - measure of how "noisy" vs "tonal" the sound is
        flatness = librosa.feature.spectral_flatness(
            y=y,
            n_fft=FeatureConfig.N_FFT,
            hop_length=FeatureConfig.HOP_LENGTH
        )
        
        # Stack all features excluding contrast
        all_features = np.vstack([
            mfcc,           # MFCC (40)
            delta_mfcc,     # Delta MFCC (40)
            delta2_mfcc,    # Delta2 MFCC (40)
            mel_spec,       # Mel spectrogram (128)
            centroid,       # Spectral centroid (1)
            rolloff,        # Spectral rolloff (1) 
            flatness        # Spectral flatness (1)
        ])
        
        # Normalize features for better training
        all_features = (all_features - np.mean(all_features, axis=1, keepdims=True)) / (np.std(all_features, axis=1, keepdims=True) + 1e-6)
        
        return all_features
        
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return None

def augment_audio(y, sr):
    """Apply multiple augmentation techniques to create varied training samples."""
    augmented = []
    
    # Original audio
    augmented.append(y)
    
    # 1. Add white noise at different levels
    noise_factor1 = 0.005
    noise_factor2 = 0.01
    noise1 = np.random.normal(0, noise_factor1, len(y))
    noise2 = np.random.normal(0, noise_factor2, len(y))
    augmented.append(y + noise1)
    augmented.append(y + noise2)
    
    # 2. Time stretching - speed up and slow down
    try:
        augmented.append(librosa.effects.time_stretch(y, rate=0.9))  # Slower
        augmented.append(librosa.effects.time_stretch(y, rate=1.1))  # Faster
    except:
        pass  # Skip if fails
    
    # 3. Pitch shifting
    try:
        augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=2))  # Pitch up
        augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-2))  # Pitch down
    except:
        pass  # Skip if fails
    
    # 4. Random time shifting
    shift = int(sr * 0.5)  # Shift by 0.5 seconds
    direction = np.random.choice([-1, 1])
    if direction == 1:
        # Shift right
        augmented.append(np.pad(y[:-shift], (shift, 0), mode='constant'))
    else:
        # Shift left
        augmented.append(np.pad(y[shift:], (0, shift), mode='constant'))
    
    return augmented

class AudioDataGenerator(tf.keras.utils.Sequence):
    """Custom data generator with improved augmentation and feature handling."""
    def __init__(self, file_paths, labels, batch_size=32, shuffle=True, augment=True):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment  # Whether to apply augmentation
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
        
        # Calculate total feature height (removed spectral contrast)
        # MFCC (40) + Delta (40) + Delta2 (40) + Mel (128) + Centroid (1) + Rolloff (1) + Flatness (1)
        feature_height = 251
        
        # Define fixed time frames (columns)
        feature_width = 130
        
        for file_path, label in zip(batch_paths, batch_labels):
            try:
                # Load and preprocess audio
                audio, sr = load_and_preprocess_audio(file_path)
                if audio is None:
                    continue
                    
                # Extract features
                features = extract_features(audio, sr)
                if features is None:
                    continue
                
                # Ensure consistent dimensions
                reshaped_features = np.zeros((feature_height, feature_width))
                
                # Get actual dimensions
                feat_rows, feat_cols = features.shape
                
                # Copy only what fits, ensuring uniform dimensions
                rows_to_use = min(feat_rows, feature_height)
                cols_to_use = min(feat_cols, feature_width)
                
                # Place features in the fixed-size array
                reshaped_features[:rows_to_use, :cols_to_use] = features[:rows_to_use, :cols_to_use]
                
                # Add to batch
                X_batch.append(reshaped_features)
                y_batch.append(label)
                
                # Apply augmentation for training data only (if enabled)
                if self.augment:
                    augmented_audio = augment_audio(audio, sr)
                    
                    # Only use a random subset of augmentations to avoid too large batches
                    if len(augmented_audio) > 3:
                        # Skip the first one (original) and select 2 random augmentations
                        aug_indices = np.random.choice(range(1, len(augmented_audio)), 2, replace=False)
                        aug_subset = [augmented_audio[i] for i in aug_indices]
                    else:
                        aug_subset = augmented_audio[1:]  # Skip original
                    
                    for aug_audio in aug_subset:
                        aug_features = extract_features(aug_audio, sr)
                        if aug_features is None:
                            continue
                        
                        # Reshape augmented features
                        aug_reshaped = np.zeros((feature_height, feature_width))
                        aug_rows, aug_cols = aug_features.shape
                        aug_rows_to_use = min(aug_rows, feature_height)
                        aug_cols_to_use = min(aug_cols, feature_width)
                        aug_reshaped[:aug_rows_to_use, :aug_cols_to_use] = aug_features[:aug_rows_to_use, :aug_cols_to_use]
                        
                        X_batch.append(aug_reshaped)
                        y_batch.append(label)
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        if not X_batch:  # If no valid features were extracted
            # Return a dummy batch with proper dimensions
            return np.zeros((1, feature_height, feature_width, 1)), np.zeros(1)
        
        # Convert to numpy arrays with explicit dtypes
        X_batch = np.array(X_batch, dtype=np.float32)
        y_batch = np.array(y_batch, dtype=np.int32)
        
        # Add channel dimension for CNN
        X_batch = X_batch[..., np.newaxis]
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def create_model(input_shape, num_classes):
    """Create improved hybrid model for emotional speech recognition with deep residual connections."""
    # Input layer
    inputs = tf.keras.Input(shape=input_shape)
    
    # Reshape for Conv1D layers - keep channel dimension
    x = layers.Reshape((-1, input_shape[0]))(inputs)
    
    # First convolutional block with residual connection
    conv1 = layers.Conv1D(128, 3, padding='same')(x)
    bn1 = layers.BatchNormalization()(conv1)
    act1 = layers.Activation('relu')(bn1)
    drop1 = layers.Dropout(0.2)(act1)
    
    # Residual block 1
    res1_conv1 = layers.Conv1D(128, 3, padding='same')(drop1)
    res1_bn1 = layers.BatchNormalization()(res1_conv1)
    res1_act1 = layers.Activation('relu')(res1_bn1)
    res1_drop1 = layers.Dropout(0.2)(res1_act1)
    
    res1_conv2 = layers.Conv1D(128, 3, padding='same')(res1_drop1)
    res1_bn2 = layers.BatchNormalization()(res1_conv2)
    res1_add = layers.Add()([drop1, res1_bn2])  # Skip connection
    res1_act2 = layers.Activation('relu')(res1_add)
    res1_pool = layers.MaxPooling1D(2)(res1_act2)
    
    # Residual block 2
    res2_conv1 = layers.Conv1D(256, 3, padding='same')(res1_pool)
    res2_bn1 = layers.BatchNormalization()(res2_conv1)
    res2_act1 = layers.Activation('relu')(res2_bn1)
    res2_drop1 = layers.Dropout(0.3)(res2_act1)
    
    res2_conv2 = layers.Conv1D(256, 3, padding='same')(res2_drop1)
    res2_bn2 = layers.BatchNormalization()(res2_conv2)
    # For skip connection, match dimensions
    res2_skip_conv = layers.Conv1D(256, 1, padding='same')(res1_pool)
    res2_add = layers.Add()([res2_skip_conv, res2_bn2])
    res2_act2 = layers.Activation('relu')(res2_add)
    res2_pool = layers.MaxPooling1D(2)(res2_act2)
    
    # Shape standardization to ensure all paths have the same output shape
    base_features = layers.Conv1D(256, 1, padding='same')(res2_pool)
    
    # Path 1: Deep CNN for high-level features
    cnn_path = layers.Conv1D(256, 3, padding='same', activation='relu')(base_features)
    cnn_path = layers.BatchNormalization()(cnn_path)
    cnn_path = layers.Dropout(0.4)(cnn_path)
    cnn_path = layers.Conv1D(256, 3, padding='same', activation='relu')(cnn_path)
    cnn_path = layers.BatchNormalization()(cnn_path)
    
    # Path 2: LSTM for temporal dynamics
    lstm_path = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(base_features)
    lstm_path = layers.Dropout(0.4)(lstm_path)
    # Make LSTM output the same size as CNN output
    lstm_path = layers.Conv1D(256, 1, padding='same')(lstm_path)
    
    # Path 3: Attention mechanism
    attention_weights = layers.Conv1D(1, 1, padding='same', activation='sigmoid')(base_features)
    attention_path = layers.Multiply()([base_features, attention_weights])
    attention_path = layers.Conv1D(256, 1, padding='same')(attention_path)
    
    # Now all three paths should have the same shape and can be concatenated
    concat = layers.Concatenate()([cnn_path, lstm_path, attention_path])
    
    # Process combined features
    x_combined = layers.Conv1D(512, 3, padding='same')(concat)
    x_combined = layers.BatchNormalization()(x_combined)
    x_combined = layers.Activation('relu')(x_combined)
    x_combined = layers.Dropout(0.4)(x_combined)
    
    # Global pooling
    x_pool = layers.GlobalAveragePooling1D()(x_combined)
    
    # Dense layers
    x_dense = layers.Dense(256, activation='relu')(x_pool)
    x_dense = layers.BatchNormalization()(x_dense)
    x_dense = layers.Dropout(0.5)(x_dense)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x_dense)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """Create detailed visualizations of training metrics with improved styling."""
    # Create a figure with subplots
    plt.figure(figsize=(20, 15))
    
    # Set a custom style for better visualizations
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Accuracy Plot with annotations
    plt.subplot(2, 2, 1)
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(train_acc) + 1)
    
    plt.plot(epochs, train_acc, 'bo-', linewidth=2, markersize=8, label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', linewidth=2, markersize=8, label='Validation Accuracy')
    
    # Annotate highest validation accuracy
    max_val_acc_epoch = np.argmax(val_acc) + 1
    max_val_acc = val_acc[max_val_acc_epoch - 1]
    plt.annotate(f'Best: {max_val_acc:.4f}',
                xy=(max_val_acc_epoch, max_val_acc),
                xytext=(max_val_acc_epoch, max_val_acc - 0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=2),
                fontsize=12, fontweight='bold')
    
    plt.title('Model Accuracy', fontsize=18, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim([0, 1.1])
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Loss Plot with annotations
    plt.subplot(2, 2, 2)
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.plot(epochs, train_loss, 'bo-', linewidth=2, markersize=8, label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', linewidth=2, markersize=8, label='Validation Loss')
    
    # Annotate lowest validation loss
    min_val_loss_epoch = np.argmin(val_loss) + 1
    min_val_loss = val_loss[min_val_loss_epoch - 1]
    plt.annotate(f'Best: {min_val_loss:.4f}',
                xy=(min_val_loss_epoch, min_val_loss),
                xytext=(min_val_loss_epoch, min_val_loss + 0.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=2),
                fontsize=12, fontweight='bold')
                
    plt.title('Model Loss', fontsize=18, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Learning Rate Plot (if available)
    if 'lr' in history.history:
        plt.subplot(2, 2, 3)
        plt.plot(epochs, history.history['lr'], 'go-', linewidth=2, markersize=8)
        plt.title('Learning Rate', fontsize=18, fontweight='bold')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Learning Rate', fontsize=14)
        plt.yscale('log')
        plt.grid(True, linestyle='--', alpha=0.7)
    
    # 4. Accuracy vs Loss Plot
    plt.subplot(2, 2, 4)
    
    # Create twin axes
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Plot accuracy and loss on separate y-axes
    acc_line, = ax1.plot(epochs, train_acc, 'b-', linewidth=2, label='Training Accuracy')
    val_acc_line, = ax1.plot(epochs, val_acc, 'b--', linewidth=2, label='Validation Accuracy')
    loss_line, = ax2.plot(epochs, train_loss, 'r-', linewidth=2, label='Training Loss')
    val_loss_line, = ax2.plot(epochs, val_loss, 'r--', linewidth=2, label='Validation Loss')
    
    # Set labels
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Accuracy', fontsize=14, color='b')
    ax2.set_ylabel('Loss', fontsize=14, color='r')
    
    # Combine legends
    lines = [acc_line, val_acc_line, loss_line, val_loss_line]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=12, loc='center right')
    
    plt.title('Accuracy vs Loss', fontsize=18, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('emotions_training_history_detailed.png', dpi=300)
    
    # Create additional metrics visualization
    plt.figure(figsize=(15, 10))
    
    # Plot convergence analysis
    plt.subplot(2, 1, 1)
    # Calculate moving average for smoother curves
    def moving_average(data, window_size=3):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    if len(train_acc) > 5:  # Only if we have enough epochs
        window = min(5, len(train_acc)//3)
        train_acc_ma = moving_average(train_acc, window)
        val_acc_ma = moving_average(val_acc, window)
        ma_epochs = range(window, len(train_acc) + 1)
        
        plt.plot(ma_epochs, train_acc_ma, 'b-', linewidth=2, label=f'Training Acc (MA{window})')
        plt.plot(ma_epochs, val_acc_ma, 'r-', linewidth=2, label=f'Validation Acc (MA{window})')
        
        # Plot gap between training and validation accuracy
        plt.fill_between(ma_epochs, train_acc_ma, val_acc_ma, 
                        color='gray', alpha=0.3, label='Generalization Gap')
    else:
        plt.plot(epochs, train_acc, 'b-', linewidth=2, label='Training Accuracy')
        plt.plot(epochs, val_acc, 'r-', linewidth=2, label='Validation Accuracy')
    
    plt.title('Convergence Analysis', fontsize=18, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot accuracy and loss distributions (final epochs)
    plt.subplot(2, 1, 2)
    # Use last 30% of training to analyze final performance
    last_n = max(1, len(train_acc) // 3)
    last_epochs = epochs[-last_n:]
    last_train_acc = train_acc[-last_n:]
    last_val_acc = val_acc[-last_n:]
    
    plt.boxplot([last_train_acc, last_val_acc], labels=['Training Acc', 'Validation Acc'])
    plt.title('Final Performance Distribution', fontsize=18, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add statistical summary as text
    plt.figtext(0.5, 0.01, 
               f"Summary (last {last_n} epochs):\n"
               f"Train Acc: mean={np.mean(last_train_acc):.4f}, std={np.std(last_train_acc):.4f}\n"
               f"Val Acc: mean={np.mean(last_val_acc):.4f}, std={np.std(last_val_acc):.4f}\n"
               f"Gap: {np.mean(last_train_acc) - np.mean(last_val_acc):.4f}",
               ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('emotions_training_convergence.png', dpi=300)
    plt.close('all')

def plot_confusion_matrix(y_true, y_pred, classes):
    """Create enhanced confusion matrix visualization with detailed metrics."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create a figure with 2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(24, 10))
    
    # Plot raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=1, linecolor='black',
                xticklabels=classes, yticklabels=classes, ax=axs[0])
    axs[0].set_title('Confusion Matrix (Counts)', fontsize=18, fontweight='bold')
    axs[0].set_xlabel('Predicted Label', fontsize=14)
    axs[0].set_ylabel('True Label', fontsize=14)
    axs[0].tick_params(axis='both', which='major', labelsize=12)
    
    # Plot normalized values (percentages)
    sns.heatmap(cm_norm, annot=True, fmt='.0%', cmap='Blues', linewidths=1, linecolor='black',
                xticklabels=classes, yticklabels=classes, ax=axs[1])
    axs[1].set_title('Confusion Matrix (Normalized)', fontsize=18, fontweight='bold')
    axs[1].set_xlabel('Predicted Label', fontsize=14)
    axs[1].set_ylabel('True Label', fontsize=14)
    axs[1].tick_params(axis='both', which='major', labelsize=12)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('emotions_confusion_matrix_detailed.png', dpi=300)
    
    # Create per-class metrics visualization
    plt.figure(figsize=(14, 10))
    
    # Calculate per-class metrics
    precision = np.zeros(len(classes))
    recall = np.zeros(len(classes))
    f1 = np.zeros(len(classes))
    
    for i in range(len(classes)):
        true_positive = cm[i, i]
        false_positive = cm[:, i].sum() - true_positive
        false_negative = cm[i, :].sum() - true_positive
        
        # Calculate metrics (handle division by zero)
        if true_positive + false_positive == 0:
            precision[i] = 0
        else:
            precision[i] = true_positive / (true_positive + false_positive)
            
        if true_positive + false_negative == 0:
            recall[i] = 0
        else:
            recall[i] = true_positive / (true_positive + false_negative)
            
        if precision[i] + recall[i] == 0:
            f1[i] = 0
        else:
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
    
    # Create grouped bar chart
    bar_width = 0.25
    index = np.arange(len(classes))
    
    plt.bar(index, precision, bar_width, label='Precision', color='#3274A1')
    plt.bar(index + bar_width, recall, bar_width, label='Recall', color='#E1812C')
    plt.bar(index + 2*bar_width, f1, bar_width, label='F1-score', color='#3A923A')
    
    plt.xlabel('Emotion Classes', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('Per-Class Performance Metrics', fontsize=18, fontweight='bold')
    plt.xticks(index + bar_width, classes, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Add class sample counts as text above each group of bars
    for i, cls in enumerate(classes):
        count = cm[i, :].sum()
        plt.text(i + bar_width, 1.05, f'n={count}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('emotions_class_metrics.png', dpi=300)
    plt.close('all')

def main():
    print("Starting Emotion Recognition Training Pipeline (Improved)...")
    
    # 1. Collect file paths
    print("\nCollecting file paths...")
    all_files = []
    all_labels = []
    class_count = {}
    
    # Check if directory exists
    if not os.path.exists(Config.DATA_PATH):
        print(f"Error: Directory '{Config.DATA_PATH}' does not exist.")
        print("Available directories in 'data/':")
        for item in os.listdir("data"):
            print(f"  - {item}")
        return
    
    # Process files for each emotion class
    for emotion_folder in os.listdir(Config.DATA_PATH):
        emotion_path = os.path.join(Config.DATA_PATH, emotion_folder)
        if os.path.isdir(emotion_path):
            print(f"Found emotion folder: {emotion_folder}")
            
            class_count[emotion_folder] = 0
            audio_files = [f for f in os.listdir(emotion_path) if f.endswith('.wav')]
            
            # Take a subset of files for faster processing
            if len(audio_files) > Config.MAX_FILES_PER_CLASS:
                audio_files = audio_files[:Config.MAX_FILES_PER_CLASS]
            
            for audio_file in audio_files:
                all_files.append(os.path.join(emotion_path, audio_file))
                all_labels.append(emotion_folder)
                class_count[emotion_folder] += 1
    
    print(f"Total files selected: {len(all_files)}")
    print("Files per class:")
    for emotion, count in class_count.items():
        print(f"  - {emotion}: {count}")
    
    if len(all_files) == 0:
        print("No audio files found. Please check the dataset structure.")
        return
    
    # 3. Prepare data for training
    print("\nPreparing data for training...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(all_labels)
    
    # Print the mapping
    print("Class mapping:")
    for i, emotion in enumerate(le.classes_):
        print(f"  - {emotion} -> {i}")
    
    # Calculate class weights to handle imbalanced data
    class_weights = {}
    total_samples = len(y_encoded)
    n_classes = len(np.unique(y_encoded))
    
    for class_idx in range(n_classes):
        class_count = np.sum(y_encoded == class_idx)
        # Formula: total_samples / (n_classes * class_count)
        weight = total_samples / (n_classes * class_count)
        class_weights[class_idx] = weight
    
    print("\nClass weights for handling imbalanced data:")
    for class_idx, weight in class_weights.items():
        print(f"  - Class {le.classes_[class_idx]}: {weight:.2f}")
    
    # Split the data with stratification
    X_train_paths, X_test_paths, y_train, y_test = train_test_split(
        all_files, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Training set: {len(X_train_paths)} files")
    print(f"Testing set: {len(X_test_paths)} files")
    
    # Create data generators
    train_generator = AudioDataGenerator(
        X_train_paths, y_train, 
        batch_size=Config.BATCH_SIZE, shuffle=True, augment=True
    )
    
    test_generator = AudioDataGenerator(
        X_test_paths, y_test, 
        batch_size=Config.BATCH_SIZE, shuffle=False, augment=False
    )
    
    # Sample batch to determine input shape
    print("\nProcessing a sample batch to determine input shape...")
    sample_batch_X, _ = train_generator[0]
    input_shape = sample_batch_X.shape[1:]
    print(f"Input shape: {input_shape}")
    
    # 4. Create and compile model
    print("\n=== Building Model ===")
    model = create_model(
        input_shape=input_shape,
        num_classes=len(le.classes_)
    )
    
    # Print model summary
    model.summary()
    
    # Learning rate warmup and decay schedule
    total_steps = len(train_generator) * Config.EPOCHS
    warmup_steps = min(1000, total_steps // 10)  # Warmup for either 1000 steps or 10% of total steps
    
    def lr_schedule(epoch, lr):
        step = epoch * len(train_generator)
        if step < warmup_steps:
            # Linear warmup phase
            return 0.0001 + step * (0.0005 - 0.0001) / warmup_steps
        else:
            # Cosine decay phase
            decay_steps = total_steps - warmup_steps
            step_in_decay = step - warmup_steps
            cosine_decay = 0.5 * (1 + np.cos(np.pi * step_in_decay / decay_steps))
            return 0.0001 + (0.0005 - 0.0001) * cosine_decay
    
    # 5. Define callbacks - enhanced with more monitoring
    best_model_path = 'best_emotions_model_advanced.h5'
    callbacks = [
        # Learning rate scheduler with warmup
        tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1),
        
        # Early stopping
        EarlyStopping(
            monitor='val_accuracy',
            patience=15, 
            restore_best_weights=True,
            verbose=1
        ),
        
        # Model checkpointing
        ModelCheckpoint(
            best_model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            update_freq='epoch'
        )
    ]
    
    # 6. Train the model
    print("\n=== Training Model ===")
    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=Config.EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,  # Add class weights to handle imbalance
        verbose=1
    )
    
    # 7. Evaluate and visualize results
    print("\n=== Evaluation ===")
    # Load best model for evaluation
    model = tf.keras.models.load_model(best_model_path)
    
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Generate predictions
    print("\nGenerating predictions for confusion matrix...")
    y_pred_probs = []
    y_true = []
    
    # Process in batches to avoid memory issues
    for i in tqdm(range(len(test_generator)), desc="Predicting"):
        X_batch, y_batch = test_generator[i]
        batch_preds = model.predict(X_batch, verbose=0)
        y_pred_probs.extend(batch_preds)
        y_true.extend(y_batch)
    
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    
    # Plot training history
    plot_training_history(history)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred_classes, le.classes_)
    
    # Print classification report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred_classes, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print(report_df)
    
    # Save the classification report
    report_df.to_csv("emotions_classification_report.csv")
    
    # Save the model
    model.save('emotions_model_advanced.h5')
    print("\nModel saved as 'emotions_model_advanced.h5'")
    
    # Free up memory
    del model
    gc.collect()

if __name__ == "__main__":
    main()