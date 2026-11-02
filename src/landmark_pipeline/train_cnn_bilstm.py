# scripts/train_cnn_bilstm.py

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    TimeDistributed,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# CONFIG
# -----------------------------

SEGMENT_ROOT = "segmented_landmarks_augmented"
TIMESTEPS = 30
FEATURES = 63
TEST_SIZE = 0.2
RANDOM_STATE = 42

# -----------------------------
# LOAD DATA
# -----------------------------

X = []
y = []

classes = sorted(os.listdir(SEGMENT_ROOT))

for cls in classes:
    cls_path = os.path.join(SEGMENT_ROOT, cls)
    if not os.path.isdir(cls_path):
        continue

    for file in os.listdir(cls_path):
        if file.endswith(".npy"):
            data = np.load(os.path.join(cls_path, file))
            X.append(data)
            y.append(cls)

X = np.array(X)
y = np.array(y)

print("Data shape:", X.shape)
print("Classes:", classes)

# -----------------------------
# ENCODE LABELS
# -----------------------------

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=TEST_SIZE,
    stratify=y_encoded,
    random_state=RANDOM_STATE
)

# -----------------------------
# MODEL: CNN + BiLSTM
# -----------------------------

model = Sequential([
    Input(shape=(TIMESTEPS, FEATURES)),  # (30, 63)

    # CNN over time dimension
    Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu",
        padding="same"
    ),
    MaxPooling1D(pool_size=2),

    Conv1D(
        filters=128,
        kernel_size=3,
        activation="relu",
        padding="same"
    ),
    MaxPooling1D(pool_size=2),

    # Temporal modeling
    Bidirectional(
        LSTM(64, return_sequences=False)
    ),

    Dropout(0.4),

    Dense(64, activation="relu"),
    Dense(len(classes), activation="softmax")
])


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# CALLBACKS
# -----------------------------

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# -----------------------------
# TRAIN
# -----------------------------

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# EVALUATION
# -----------------------------

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy (CNN + BiLSTM): {test_acc:.4f}")

y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_labels,
        target_names=le.classes_
    )
)

# -----------------------------
# SAVE MODEL
# -----------------------------

os.makedirs("models", exist_ok=True)
model.save("models/cnn_bilstm_baseline.keras")

print("\nModel saved to models/cnn_bilstm_baseline.keras")
