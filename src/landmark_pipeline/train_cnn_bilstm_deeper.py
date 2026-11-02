# scripts/train_cnn_bilstm_deeper.py

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
    BatchNormalization,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# -----------------------------
# CONFIG
# -----------------------------

DATA_ROOT = "segmented_landmarks_augmented"
TIMESTEPS = 30
FEATURES = 63
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 16
EPOCHS = 150

# -----------------------------
# LOAD DATA
# -----------------------------

X, y = [], []

classes = sorted(os.listdir(DATA_ROOT))

for cls in classes:
    cls_path = os.path.join(DATA_ROOT, cls)
    if not os.path.isdir(cls_path):
        continue

    for file in os.listdir(cls_path):
        if file.endswith(".npy"):
            X.append(np.load(os.path.join(cls_path, file)))
            y.append(cls)

X = np.array(X)
y = np.array(y)

print("Data shape:", X.shape)
print("Total samples:", len(X))
print("Classes:", classes)

# -----------------------------
# LABEL ENCODING
# -----------------------------

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# -----------------------------
# TRAIN–TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=TEST_SIZE,
    stratify=y_encoded,
    random_state=RANDOM_STATE
)

# -----------------------------
# MODEL: DEEPER CNN + BiLSTM
# -----------------------------

model = Sequential([
    Input(shape=(TIMESTEPS, FEATURES)),

    # -------- CNN BLOCK 1 --------
    Conv1D(64, kernel_size=3, padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    # -------- CNN BLOCK 2 --------
    Conv1D(128, kernel_size=3, padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    # -------- CNN BLOCK 3 (deeper) --------
    Conv1D(256, kernel_size=3, padding="same", activation="relu"),
    BatchNormalization(),

    # -------- TEMPORAL MODELING --------
    Bidirectional(
        LSTM(128, return_sequences=True)
    ),
    Bidirectional(
        LSTM(64, return_sequences=False)
    ),

    Dropout(0.5),

    Dense(128, activation="relu"),
    Dropout(0.3),

    Dense(len(classes), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# CALLBACKS
# -----------------------------

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=7,
    min_lr=1e-5,
    verbose=1
)

# -----------------------------
# TRAIN
# -----------------------------

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# -----------------------------
# EVALUATION
# -----------------------------

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy (Deeper CNN + BiLSTM): {test_acc:.4f}")

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
model.save("models/cnn_bilstm_deeper.keras")

print("\nModel saved to models/cnn_bilstm_deeper.keras")
