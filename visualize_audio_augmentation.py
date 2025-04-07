import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import random
import glob
from matplotlib.gridspec import GridSpec
import seaborn as sns
from IPython.display import Audio
import warnings
warnings.filterwarnings("ignore")

# Functions for audio augmentation
def noise(data, noise_factor=0.005):
    """Add random noise to sound"""
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def shift(data, sr, shift_factor=0.2, direction='right'):
    """Shift the audio data"""
    shift = int(sr * shift_factor)
    if direction == 'right':
        augmented_data = np.roll(data, shift)
    else:
        augmented_data = np.roll(data, -shift)
    return augmented_data

def pitch(data, sr, pitch_factor=0.7):
    """Pitch shift the audio data"""
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=pitch_factor)

def stretch(data, rate=1.1):
    """Stretch the audio data"""
    return librosa.effects.time_stretch(data, rate=rate)

def extract_mfcc(data, sr):
    """Extract MFCC features"""
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
    return mfccs

def plot_audio_augmentations(emotion_name, audio_file):
    """
    Plot original audio and augmented versions for a given emotion
    """
    # Load the audio file
    data, sr = librosa.load(audio_file, sr=22050)
    
    # Create augmented versions
    data_noise = noise(data, noise_factor=0.01)
    data_shift = shift(data, sr, shift_factor=0.2, direction='right')
    data_pitch = pitch(data, sr, pitch_factor=0.7)
    data_stretch = stretch(data, rate=0.9)
    
    # Extract MFCCs
    mfccs_original = extract_mfcc(data, sr)
    mfccs_noise = extract_mfcc(data_noise, sr)
    mfccs_shift = extract_mfcc(data_shift, sr)
    mfccs_pitch = extract_mfcc(data_pitch, sr)
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(5, 2, height_ratios=[1, 1, 1, 1, 1.5])
    
    # Plot title
    fig.suptitle(f'Audio Augmentation Visualization - {emotion_name.capitalize()}', fontsize=16)
    
    # Plot original audio
    ax1 = fig.add_subplot(gs[0, 0])
    librosa.display.waveshow(data, sr=sr, ax=ax1, color='salmon')
    ax1.set_title(f'Original Audio ({emotion_name})')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    
    # Plot noise augmentation
    ax2 = fig.add_subplot(gs[0, 1])
    librosa.display.waveshow(data_noise, sr=sr, ax=ax2, color='orangered')
    ax2.set_title('Noise Augmentation')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amplitude')
    
    # Plot shift augmentation
    ax3 = fig.add_subplot(gs[1, 0])
    librosa.display.waveshow(data_shift, sr=sr, ax=ax3, color='orangered')
    ax3.set_title('Time Shift Augmentation')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Amplitude')
    
    # Plot pitch augmentation
    ax4 = fig.add_subplot(gs[1, 1])
    librosa.display.waveshow(data_pitch, sr=sr, ax=ax4, color='orangered')
    ax4.set_title('Pitch Shift Augmentation')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Amplitude')
    
    # Plot stretch augmentation
    ax5 = fig.add_subplot(gs[2, 0])
    librosa.display.waveshow(data_stretch, sr=sr, ax=ax5, color='orangered')
    ax5.set_title('Time Stretch Augmentation')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Amplitude')
    
    # Plot spectrogram
    ax6 = fig.add_subplot(gs[2, 1])
    D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax6, cmap='viridis')
    ax6.set_title('Spectrogram')
    plt.colorbar(ax6.collections[0], ax=ax6, format='%+2.0f dB')
    
    # Plot MFCCs comparison
    ax7 = fig.add_subplot(gs[3, :])
    mfcc_diff = np.abs(mfccs_original - mfccs_noise)
    librosa.display.specshow(mfcc_diff, sr=sr, ax=ax7, cmap='coolwarm')
    ax7.set_title('MFCC Difference (Original vs Noise)')
    plt.colorbar(ax7.collections[0], ax=ax7)
    
    # Plot augmentation parameters
    ax8 = fig.add_subplot(gs[4, :])
    ax8.set_title('Augmentation Parameters')
    ax8.set_xlim(0, 3.5)
    ax8.set_ylim(0, 1)
    ax8.set_yticks([])
    
    # Noise factor slider visualization
    ax8.text(0.05, 0.8, 'Noise Factor', fontsize=12)
    ax8.plot([0, 3.5], [0.7, 0.7], 'gray', alpha=0.5)
    ax8.plot(0.01 * 350, 0.7, 'bo', markersize=12)
    ax8.text(3.6, 0.7, '0.01', fontsize=10)
    
    # Shift factor slider visualization
    ax8.text(0.05, 0.5, 'Shift Factor', fontsize=12)
    ax8.plot([0, 3.5], [0.4, 0.4], 'gray', alpha=0.5)
    ax8.plot(0.2 * 350 / 3.5, 0.4, 'bo', markersize=12)
    ax8.text(3.6, 0.4, '0.2', fontsize=10)
    
    # Pitch steps slider visualization
    ax8.text(0.05, 0.2, 'Pitch Steps', fontsize=12)
    ax8.plot([0, 3.5], [0.1, 0.1], 'gray', alpha=0.5)
    ax8.plot(0.7 * 350 / 3.5, 0.1, 'bo', markersize=12)
    ax8.text(3.6, 0.1, '0.7', fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the plot
    output_dir = 'augmentation_plots'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/{emotion_name}_augmentation.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_emotion_features(emotion_dir, emotion_name):
    """
    Generate feature visualizations for a specific emotion
    """
    # Get all audio files for this emotion
    audio_files = glob.glob(os.path.join(emotion_dir, "*.wav"))
    
    if not audio_files:
        print(f"No audio files found for emotion: {emotion_name}")
        return
    
    # Select a random file for visualization
    sample_file = random.choice(audio_files)
    print(f"Visualizing augmentations for {emotion_name} using file: {os.path.basename(sample_file)}")
    
    # Generate plots
    plot_audio_augmentations(emotion_name, sample_file)

def visualize_all_emotions():
    """
    Create visualizations for all emotions in the dataset
    """
    # Data directory with emotions
    data_directory = "data/Emotions"
    
    # Check if space in directory name
    if not os.path.exists(data_directory):
        data_directory = "data/Emotions "
        if not os.path.exists(data_directory):
            print("Error: Cannot find the Emotions directory. Please check the path.")
            return
    
    print(f"Searching for emotion folders in: {data_directory}")
    
    # Process each emotion folder
    emotion_folders = []
    for item in os.listdir(data_directory):
        folder_path = os.path.join(data_directory, item)
        if os.path.isdir(folder_path) and not item.startswith('.'):
            emotion_folders.append((folder_path, item))
    
    if not emotion_folders:
        print("No emotion folders found!")
        return
    
    print(f"Found {len(emotion_folders)} emotion folders: {[name for _, name in emotion_folders]}")
    
    # Process each emotion
    for emotion_dir, emotion_name in emotion_folders:
        visualize_emotion_features(emotion_dir, emotion_name)
    
    print(f"Visualization complete! Images saved to 'augmentation_plots' directory")

def create_combined_visualization():
    """
    Create a combined plot comparing MFCCs across different emotions
    """
    # Data directory with emotions
    data_directory = "data/Emotions"
    if not os.path.exists(data_directory):
        data_directory = "data/Emotions "
        if not os.path.exists(data_directory):
            print("Error: Cannot find the Emotions directory")
            return
    
    emotion_samples = {}
    
    # Get sample audio for each emotion
    for emotion in os.listdir(data_directory):
        emotion_dir = os.path.join(data_directory, emotion)
        if os.path.isdir(emotion_dir) and not emotion.startswith('.'):
            audio_files = glob.glob(os.path.join(emotion_dir, "*.wav"))
            if audio_files:
                emotion_samples[emotion] = random.choice(audio_files)
    
    if not emotion_samples:
        print("No emotion samples found!")
        return
    
    # Create a figure for MFCC comparison
    n_emotions = len(emotion_samples)
    plt.figure(figsize=(15, n_emotions * 3))
    
    for i, (emotion, audio_file) in enumerate(emotion_samples.items()):
        # Load audio and extract MFCCs
        data, sr = librosa.load(audio_file, sr=22050)
        mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
        
        # Plot MFCCs
        plt.subplot(n_emotions, 1, i+1)
        librosa.display.specshow(mfccs, sr=sr, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'MFCC - {emotion}')
        plt.tight_layout()
    
    # Save the figure
    output_dir = 'augmentation_plots'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/emotion_mfcc_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created combined MFCC visualization")

def create_feature_extraction_visualization():
    """
    Create a visualization showing feature extraction process
    """
    # Data directory with emotions
    data_directory = "data/Emotions"
    if not os.path.exists(data_directory):
        data_directory = "data/Emotions "
        if not os.path.exists(data_directory):
            print("Error: Cannot find the Emotions directory")
            return
    
    # Find all audio files
    all_files = []
    for root, _, files in os.walk(data_directory):
        for file in files:
            if file.endswith('.wav'):
                all_files.append(os.path.join(root, file))
    
    if not all_files:
        print("No audio files found!")
        return
    
    # Select a random file
    sample_file = random.choice(all_files)
    emotion = os.path.basename(os.path.dirname(sample_file))
    
    # Load audio
    data, sr = librosa.load(sample_file, sr=22050)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot waveform
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(data, sr=sr, color='blue')
    plt.title(f'Waveform ({emotion})')
    
    # Plot spectrogram
    plt.subplot(3, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    
    # Plot MFCCs
    plt.subplot(3, 1, 3)
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCCs')
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = 'augmentation_plots'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/feature_extraction_process.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created feature extraction visualization")

if __name__ == "__main__":
    print("Starting audio augmentation visualization...")
    
    # Create output directory
    os.makedirs('augmentation_plots', exist_ok=True)
    
    # Visualize augmentations for each emotion
    visualize_all_emotions()
    
    # Create additional visualizations
    create_combined_visualization()
    create_feature_extraction_visualization()
    
    print("All visualizations complete!") 