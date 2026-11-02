import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# CONFIG
# -----------------------------
DATA_ROOT = "augmented_landmarks"
SEQ_LEN = 30
FEATURES = 63
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 100
BATCH_SIZE = 16

# -----------------------------
# LOAD DATA
# -----------------------------
X, y = [], []

for class_name in sorted(os.listdir(DATA_ROOT)):
    class_path = os.path.join(DATA_ROOT, class_name)
    if not os.path.isdir(class_path):
        continue

    for file in os.listdir(class_path):
        if file.endswith(".npy"):
            seq = np.load(os.path.join(class_path, file))

            # Safety check
            if seq.shape != (SEQ_LEN, FEATURES):
                continue

            X.append(seq)
            y.append(class_name)

X = np.array(X)
y = np.array(y)

print("Data shape:", X.shape)
print("Classes:", sorted(set(y)))

# -----------------------------
# LABEL ENCODING
# -----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=TEST_SIZE,
    stratify=y_encoded,
    random_state=RANDOM_STATE
)

# -----------------------------
# MODEL
# -----------------------------
model = Sequential([
    Input(shape=(SEQ_LEN, FEATURES)),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# EARLY STOPPING
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
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# EVALUATION
# -----------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy (Augmented): {test_acc:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    cmap="Blues"
)
plt.title("Confusion Matrix (Augmented Data)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# -----------------------------
# TRAINING CURVES
# -----------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Accuracy Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss Curve")
plt.legend()

plt.tight_layout()
plt.show()

# -----------------------------
# SAVE MODEL
# -----------------------------
os.makedirs("models", exist_ok=True)
model.save("models/lstm_augmented.keras")
print("\nModel saved to models/lstm_augmented.keras")
