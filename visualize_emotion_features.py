import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import glob
import random
import seaborn as sns
from matplotlib.gridspec import GridSpec
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")

def extract_features(file_path):
    """Extract audio features from a file"""
    try:
        # Load the audio file
        data, sr = librosa.load(file_path, sr=22050)
        
        # Extract features
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=data, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
        
        # Concatenate features
        features = np.concatenate((mfccs, chroma, mel))
        return features, data, sr
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None, None, None

def collect_emotion_samples(limit_per_emotion=10):
    """Collect audio samples for each emotion"""
    # Data directory with emotions
    data_directory = "data/Emotions"
    if not os.path.exists(data_directory):
        data_directory = "data/Emotions "
        if not os.path.exists(data_directory):
            print("Error: Cannot find the Emotions directory. Please check the path.")
            return {}, []
    
    emotion_samples = {}
    all_features = []
    all_emotions = []
    
    # Process each emotion folder
    for emotion in os.listdir(data_directory):
        emotion_dir = os.path.join(data_directory, emotion)
        if os.path.isdir(emotion_dir) and not emotion.startswith('.'):
            # Get all audio files for this emotion
            audio_files = glob.glob(os.path.join(emotion_dir, "*.wav"))
            
            if audio_files:
                # Limit the number of files to process
                if limit_per_emotion and len(audio_files) > limit_per_emotion:
                    audio_files = random.sample(audio_files, limit_per_emotion)
                
                emotion_samples[emotion] = audio_files
                
                # Extract features for each file
                for file in audio_files:
                    features, _, _ = extract_features(file)
                    if features is not None:
                        all_features.append(features)
                        all_emotions.append(emotion)
    
    return emotion_samples, all_features, all_emotions

def visualize_emotion_waveforms(emotion_samples):
    """Create waveform comparisons for different emotions"""
    plt.figure(figsize=(15, 10))
    
    # Display one sample waveform for each emotion
    for i, (emotion, files) in enumerate(emotion_samples.items()):
        if files:  # Make sure there are files for this emotion
            # Select a random file
            file = random.choice(files)
            
            # Load audio data
            data, sr = librosa.load(file, sr=22050)
            
            # Plot waveform
            plt.subplot(len(emotion_samples), 1, i+1)
            librosa.display.waveshow(data, sr=sr, color=plt.cm.tab10(i % 10))
            plt.title(f'Waveform: {emotion}')
            plt.ylim(-0.5, 0.5)  # Consistent y-axis scale
            
            # Only show x-axis label for the bottom plot
            if i == len(emotion_samples) - 1:
                plt.xlabel('Time (s)')
            
            plt.ylabel('Amplitude')
    
    plt.tight_layout()
    output_dir = 'emotion_features'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/emotion_waveform_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created emotion waveform comparison chart")

def visualize_mfcc_comparison(emotion_samples):
    """Create MFCC comparison for different emotions"""
    plt.figure(figsize=(15, 12))
    
    # Display MFCC for each emotion
    for i, (emotion, files) in enumerate(emotion_samples.items()):
        if files:  # Make sure there are files for this emotion
            # Select a random file
            file = random.choice(files)
            
            # Load audio data and extract MFCCs
            data, sr = librosa.load(file, sr=22050)
            mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
            
            # Plot MFCCs
            plt.subplot(len(emotion_samples), 1, i+1)
            img = librosa.display.specshow(mfccs, sr=sr, x_axis='time')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'MFCC: {emotion}')
            
            # Only show x-axis label for the bottom plot
            if i == len(emotion_samples) - 1:
                plt.xlabel('Time (s)')
            
            plt.ylabel('MFCC Coefficients')
    
    plt.tight_layout()
    output_dir = 'emotion_features'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/emotion_mfcc_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created emotion MFCC comparison chart")

def visualize_feature_distributions(all_features, all_emotions):
    """Visualize feature distributions across emotions"""
    # Convert to numpy arrays
    X = np.array(all_features)
    y = np.array(all_emotions)
    
    # Create PCA and t-SNE visualizations
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2, random_state=42)
    
    # Apply dimensionality reduction
    pca_result = pca.fit_transform(X)
    tsne_result = tsne.fit_transform(X)
    
    # Get unique emotions and assign colors
    unique_emotions = list(set(all_emotions))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_emotions)))
    
    # Create figure
    plt.figure(figsize=(15, 7))
    
    # Plot PCA results
    plt.subplot(1, 2, 1)
    for i, emotion in enumerate(unique_emotions):
        indices = y == emotion
        plt.scatter(pca_result[indices, 0], pca_result[indices, 1], 
                   c=[colors[i]], label=emotion, alpha=0.7)
    plt.title('PCA: Emotion Feature Distribution')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    
    # Plot t-SNE results
    plt.subplot(1, 2, 2)
    for i, emotion in enumerate(unique_emotions):
        indices = y == emotion
        plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], 
                   c=[colors[i]], label=emotion, alpha=0.7)
    plt.title('t-SNE: Emotion Feature Distribution')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.legend()
    
    plt.tight_layout()
    output_dir = 'emotion_features'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/emotion_feature_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created emotion feature distribution chart")

def create_feature_heatmap(all_features, all_emotions, n_features=20):
    """Create a heatmap of the most important features for each emotion"""
    # Convert to pandas DataFrame for easier analysis
    df = pd.DataFrame(all_features)
    df['emotion'] = all_emotions
    
    # Calculate mean feature values for each emotion
    emotion_means = df.groupby('emotion').mean()
    
    # Select subset of features for visualization
    if emotion_means.shape[1] > n_features:
        emotion_means = emotion_means.iloc[:, :n_features]
    
    # Create heatmap
    plt.figure(figsize=(15, 8))
    sns.heatmap(emotion_means, cmap='viridis', annot=False, fmt=".2f")
    plt.title('Mean Feature Values by Emotion')
    plt.xlabel('Feature Index')
    plt.ylabel('Emotion')
    
    plt.tight_layout()
    output_dir = 'emotion_features'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/emotion_feature_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created emotion feature heatmap")

def visualize_feature_boxplots(all_features, all_emotions):
    """Create boxplots of key features across emotions"""
    # Convert to pandas DataFrame
    df = pd.DataFrame(all_features)
    df['emotion'] = all_emotions
    
    # Select a few key features for visualization
    key_features = [0, 1, 2, 3, 4]  # First few MFCC coefficients
    
    plt.figure(figsize=(15, 10))
    
    for i, feature_idx in enumerate(key_features):
        plt.subplot(len(key_features), 1, i+1)
        sns.boxplot(x='emotion', y=feature_idx, data=df)
        plt.title(f'Feature {feature_idx} Distribution by Emotion')
        plt.ylabel(f'Feature {feature_idx} Value')
        plt.xlabel('')  # Only show x-axis label on bottom plot
    
    plt.xlabel('Emotion')
    plt.tight_layout()
    output_dir = 'emotion_features'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/emotion_feature_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created emotion feature boxplots")

def visualize_spectrograms(emotion_samples):
    """Create spectrogram comparisons for different emotions"""
    plt.figure(figsize=(15, 12))
    
    # Display spectrogram for each emotion
    for i, (emotion, files) in enumerate(emotion_samples.items()):
        if files:  # Make sure there are files for this emotion
            # Select a random file
            file = random.choice(files)
            
            # Load audio data
            data, sr = librosa.load(file, sr=22050)
            
            # Compute spectrogram
            D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
            
            # Plot spectrogram
            plt.subplot(len(emotion_samples), 1, i+1)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Spectrogram: {emotion}')
            
            # Only show x-axis label for the bottom plot
            if i == len(emotion_samples) - 1:
                plt.xlabel('Time (s)')
            
            plt.ylabel('Frequency (Hz)')
    
    plt.tight_layout()
    output_dir = 'emotion_features'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/emotion_spectrogram_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created emotion spectrogram comparison chart")

def main():
    print("Starting emotion feature visualization...")
    
    # Create output directory
    output_dir = 'emotion_features'
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect audio samples and extract features
    print("Collecting audio samples and extracting features...")
    emotion_samples, all_features, all_emotions = collect_emotion_samples(limit_per_emotion=5)
    
    if not emotion_samples:
        print("No emotion samples found. Please check your data directory.")
        return
    
    print(f"Found {len(emotion_samples)} emotions: {list(emotion_samples.keys())}")
    print(f"Extracted features from {len(all_features)} audio files")
    
    # Create different visualizations
    print("Creating visualizations...")
    visualize_emotion_waveforms(emotion_samples)
    visualize_mfcc_comparison(emotion_samples)
    visualize_spectrograms(emotion_samples)
    
    if all_features:
        visualize_feature_distributions(all_features, all_emotions)
        create_feature_heatmap(all_features, all_emotions)
        visualize_feature_boxplots(all_features, all_emotions)
    
    print(f"All visualizations complete! Images saved to '{output_dir}' directory")

if __name__ == "__main__":
    main() 