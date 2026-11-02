# scripts/train_cnn_bilstm_angles.py

import os
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# CONFIG
# -----------------------------

SEGMENT_ROOT = "segmented_landmarks_with_angles"
TIMESTEPS = 30
FEATURES = 84 # 63 landmarks + angle features
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 16
EPOCHS = 100

# -----------------------------
# LOAD DATA
# -----------------------------

X, y = [], []

classes = sorted(os.listdir(SEGMENT_ROOT))

for cls in classes:
    cls_path = os.path.join(SEGMENT_ROOT, cls)
    if not os.path.isdir(cls_path):
        continue

    for file in os.listdir(cls_path):
        if file.endswith(".npy"):
            X.append(np.load(os.path.join(cls_path, file)))
            y.append(cls)

X = np.array(X)
y = np.array(y)

print("Data shape:", X.shape)
print("Classes:", classes)

# -----------------------------
# LABEL ENCODING
# -----------------------------

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# -----------------------------
# TRAIN-TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=TEST_SIZE,
    stratify=y_encoded,
    random_state=RANDOM_STATE
)

# -----------------------------
# MODEL: CNN + BiLSTM + Angles
# -----------------------------

model = Sequential([
    Input(shape=(TIMESTEPS, FEATURES)),

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
    patience=12,
    restore_best_weights=True
)

# -----------------------------
# TRAIN
# -----------------------------

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# EVALUATION
# -----------------------------

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy (CNN + BiLSTM + Angles): {test_acc:.4f}")

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
model.save("models/cnn_bilstm_angles.keras")

print("\nModel saved to models/cnn_bilstm_angles.keras")
