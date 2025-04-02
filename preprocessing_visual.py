#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speech Emotion Recognition - Feature Extraction & Augmentation Visualizations
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.widgets import Slider, Button
import matplotlib.gridspec as gridspec

# Set style for plots
plt.style.use('ggplot')

# Create visual_plots directory if it doesn't exist
if not os.path.exists('visual_plots'):
    os.makedirs('visual_plots')

# Emotion dictionary
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

def load_sample_audio(data_directory="data/ravdess"):
    """Load one audio file per emotion for visualization"""
    sample_audio = {}
    
    for file in glob.glob(data_directory + "/Actor_*/*.wav"):
        emotion_code = os.path.basename(file).split("-")[2]
        emotion = emotions[emotion_code]
        
        if emotion not in sample_audio:
            data, sr = librosa.load(file)
            sample_audio[emotion] = (data, sr)
            
        if len(sample_audio) == len(emotions):
            break
            
    return sample_audio

def save_plot(fig, filename):
    """Helper function to save plots consistently"""
    path = os.path.join('visual_plots', filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")

def plot_waveform(data, sr, emotion):
    """Plot raw waveform"""
    fig = plt.figure(figsize=(12, 4))
    librosa.display.waveshow(data, sr=sr, alpha=0.5)
    plt.title(f"Raw Audio Waveform (Emotion: {emotion})")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.tight_layout()
    save_plot(fig, f"waveform_{emotion}.png")

def plot_spectrogram(data, sr, emotion):
    """Plot spectrogram"""
    fig = plt.figure(figsize=(12, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectrogram (Emotion: {emotion})")
    plt.tight_layout()
    save_plot(fig, f"spectrogram_{emotion}.png")

def plot_mel_spectrogram(data, sr, emotion):
    """Plot Mel spectrogram"""
    fig = plt.figure(figsize=(12, 4))
    S = librosa.feature.melspectrogram(y=data, sr=sr)
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Mel Spectrogram (Emotion: {emotion})")
    plt.tight_layout()
    save_plot(fig, f"mel_spectrogram_{emotion}.png")

def plot_mfccs(data, sr, emotion):
    """Plot MFCCs"""
    fig = plt.figure(figsize=(12, 4))
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title(f"MFCCs (Emotion: {emotion})")
    plt.tight_layout()
    save_plot(fig, f"mfccs_{emotion}.png")

def noise(data, noise_factor=0.01):
    """Add random noise to audio"""
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data.astype(type(data[0]))

def shift(data, sr, shift_factor=0.2, direction='right'):
    """Shift audio in time"""
    shift = int(sr * shift_factor)
    return np.roll(data, shift) if direction == 'right' else np.roll(data, -shift)

def pitch_shift(data, sr, pitch_factor=0.7):
    """Pitch shift audio"""
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=pitch_factor)

def interactive_augmentation_demo(data, sr, emotion):
    """Interactive demo of augmentation effects"""
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(3, 2, figure=fig)
    
    # Original plot
    ax1 = fig.add_subplot(gs[0, 0])
    librosa.display.waveshow(data, sr=sr, alpha=0.5, ax=ax1)
    ax1.set_title(f"Original Audio ({emotion})")
    
    # Noise plot
    ax2 = fig.add_subplot(gs[0, 1])
    noise_data = noise(data)
    line_noise, = ax2.plot(np.arange(len(noise_data))/sr, noise_data)
    ax2.set_title("Noise Augmentation")
    
    # Shift plot
    ax3 = fig.add_subplot(gs[1, 0])
    shift_data = shift(data, sr)
    line_shift, = ax3.plot(np.arange(len(shift_data))/sr, shift_data)
    ax3.set_title("Time Shift Augmentation")
    
    # Pitch plot
    ax4 = fig.add_subplot(gs[1, 1])
    pitch_data = pitch_shift(data, sr)
    line_pitch, = ax4.plot(np.arange(len(pitch_data))/sr, pitch_data)
    ax4.set_title("Pitch Shift Augmentation")
    
    # MFCC comparison
    ax5 = fig.add_subplot(gs[2, :])
    mfcc_orig = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
    mfcc_noise = librosa.feature.mfcc(y=noise_data, sr=sr, n_mfcc=13)
    librosa.display.specshow(mfcc_orig, sr=sr, x_axis='time', ax=ax5)
    ax5.set_title("MFCC Comparison (Original vs Augmented)")
    
    plt.tight_layout()
    
    # Add sliders
    ax_noise = plt.axes([0.25, 0.05, 0.5, 0.02])
    noise_slider = Slider(ax_noise, 'Noise Factor', 0.0, 0.1, valinit=0.01)
    
    ax_shift = plt.axes([0.25, 0.02, 0.5, 0.02])
    shift_slider = Slider(ax_shift, 'Shift Factor', 0.0, 0.5, valinit=0.2)
    
    ax_pitch = plt.axes([0.25, -0.01, 0.5, 0.02])
    pitch_slider = Slider(ax_pitch, 'Pitch Steps', -2.0, 2.0, valinit=0.7)
    
    def update(val):
        # Update noise
        new_noise = noise(data, noise_slider.val)
        line_noise.set_ydata(new_noise)
        
        # Update shift
        new_shift = shift(data, sr, shift_slider.val)
        line_shift.set_ydata(new_shift)
        
        # Update pitch
        new_pitch = pitch_shift(data, sr, pitch_slider.val)
        line_pitch.set_ydata(new_pitch)
        
        # Update MFCCs
        mfcc_noise = librosa.feature.mfcc(y=new_noise, sr=sr, n_mfcc=13)
        ax5.clear()
        librosa.display.specshow(mfcc_orig, sr=sr, x_axis='time', ax=ax5)
        ax5.set_title("MFCC Comparison (Original vs Augmented)")
        
        fig.canvas.draw_idle()
    
    noise_slider.on_changed(update)
    shift_slider.on_changed(update)
    pitch_slider.on_changed(update)
    
    plt.show()
    save_plot(fig, f"interactive_augmentation_{emotion}.png")

def main():
    # Load sample audio files
    print("Loading sample audio files...")
    sample_audio = load_sample_audio()
    
    # Generate visualizations for each emotion
    for emotion, (data, sr) in sample_audio.items():
        print(f"\nGenerating visualizations for {emotion}...")
        
        # Basic visualizations
        plot_waveform(data, sr, emotion)
        plot_spectrogram(data, sr, emotion)
        plot_mel_spectrogram(data, sr, emotion)
        plot_mfccs(data, sr, emotion)
        
        # Interactive augmentation demo
        interactive_augmentation_demo(data, sr, emotion)
    
    print("\nAll visualizations saved to 'visual_plots' folder!")

if __name__ == "__main__":
    main()