#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speech Emotion Recognition - Model Validation Script (Fixed for Keras)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit, KFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.callbacks import EarlyStopping
import librosa
import os
import glob

# Load your saved model and data
print("Loading model and data...")
model = load_model('speech_emotion_recognition_model.h5')
X = np.load('X.npy')
y = np.load('y.npy', allow_pickle=True)

# -------------------------------------------------------------------
# 1. Check for Data Leakage (Speaker Independence)
# -------------------------------------------------------------------
print("\n🔍 Checking for Data Leakage...")

# Get list of all audio files in the same order as X and y
audio_files = sorted(glob.glob("data/ravdess/Actor_*/*.wav"))

# Verify we have matching number of samples
if len(audio_files) != len(X):
    print(f"Warning: {len(audio_files)} audio files found but {len(X)} feature vectors")
    print("Using only the first matching set...")
    min_length = min(len(audio_files), len(X))
    audio_files = audio_files[:min_length]
    X = X[:min_length]
    y = y[:min_length]

# Extract actor IDs from RAVDESS filenames (format: 03-01-01-01-01-01-01.wav)
def get_actor_id(filename):
    """Extract actor ID from RAVDESS filename"""
    return int(os.path.basename(filename).split("-")[-1].split(".")[0])

actor_ids = np.array([get_actor_id(f) for f in audio_files])

# Verify all arrays have same length
assert len(X) == len(y) == len(actor_ids), "Input arrays have different lengths!"

# GroupShuffleSplit to ensure speaker-independent split
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=actor_ids))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
train_actors, test_actors = actor_ids[train_idx], actor_ids[test_idx]

# Check for overlapping actors
overlap = set(train_actors).intersection(set(test_actors))
print(f"Train Actors: {len(set(train_actors))} | Test Actors: {len(set(test_actors))}")
print(f"Overlapping Actors: {'✅ None' if not overlap else '❌ ' + str(overlap)}")

# -------------------------------------------------------------------
# 2. Manual K-Fold Cross-Validation (for Keras models)
# -------------------------------------------------------------------
print("\n📊 Running Manual 5-Fold Cross-Validation...")

# Reshape for CNN and encode labels
X_reshaped = np.expand_dims(X, axis=2)
y_encoded = LabelEncoder().fit_transform(y)

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Manual K-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_reshaped)):
    print(f"\nFold {fold+1}/5")
    
    # Create new model instance
    fold_model = clone_model(model)
    fold_model.compile(optimizer='adam', 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
    
    # Train
    history = fold_model.fit(
        X_reshaped[train_idx], y_encoded[train_idx],
        validation_data=(X_reshaped[val_idx], y_encoded[val_idx]),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Evaluate
    _, accuracy = fold_model.evaluate(X_reshaped[val_idx], y_encoded[val_idx], verbose=0)
    cv_scores.append(accuracy)
    print(f"Fold {fold+1} Accuracy: {accuracy:.4f}")

print(f"\nCV Scores: {cv_scores}")
print(f"Mean CV Accuracy: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")

# -------------------------------------------------------------------
# 3. Confusion Matrix Analysis
# -------------------------------------------------------------------
print("\n📈 Evaluating Test Set Performance...")

# Predict on test set
X_test_cnn = np.expand_dims(X_test, axis=2)
y_pred = np.argmax(model.predict(X_test_cnn), axis=1)
y_test_encoded = LabelEncoder().fit_transform(y_test)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred, target_names=np.unique(y)))

# Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test_encoded, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Confusion Matrix (Counts)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig('confusion_matrix_validation.png')
print("\nConfusion matrix saved to 'confusion_matrix_validation.png'")

# -------------------------------------------------------------------
# 4. Class Distribution Check
# -------------------------------------------------------------------
print("\n📦 Checking Class Distribution...")

train_counts = pd.Series(y_train).value_counts()
test_counts = pd.Series(y_test).value_counts()

print("\nTrain Set Distribution:")
print(train_counts)
print("\nTest Set Distribution:")
print(test_counts)

# Plot distributions
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
train_counts.plot(kind='bar')
plt.title("Train Set Class Distribution")
plt.subplot(1, 2, 2)
test_counts.plot(kind='bar')
plt.title("Test Set Class Distribution")
plt.tight_layout()
plt.savefig('class_distributions.png')
print("Class distributions saved to 'class_distributions.png'")

print("\n✅ Validation Complete!")
print("Key outputs saved:")
print("- confusion_matrix_validation.png")
print("- class_distributions.png")
print("\nShare these results for further analysis.")