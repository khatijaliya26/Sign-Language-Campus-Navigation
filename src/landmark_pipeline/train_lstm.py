import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "segmented_landmarks"))

# -----------------------------
# LOAD DATA
# -----------------------------
X, y = [], []

for block_name in os.listdir(DATA_ROOT):
    block_path = os.path.join(DATA_ROOT, block_name)
    if not os.path.isdir(block_path):
        continue

    for file in os.listdir(block_path):
        if file.endswith(".npy"):
            X.append(np.load(os.path.join(block_path, file)))
            y.append(block_name)

X = np.array(X)
y = np.array(y)

print("Data shape:", X.shape)

# -----------------------------
# LABEL ENCODING
# -----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_onehot = to_categorical(y_encoded)

class_names = le.classes_
print("Classes:", class_names)

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test, y_train_lbl, y_test_lbl = train_test_split(
    X, y_onehot, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# -----------------------------
# MODEL
# -----------------------------
model = Sequential([
    Bidirectional(LSTM(64), input_shape=(30, 63)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# TRAIN
# -----------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=16
)

# -----------------------------
# EVALUATION
# -----------------------------
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Accuracy
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------
print("\nClassification Report:")
print(classification_report(y_test_lbl, y_pred, target_names=class_names))

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_test_lbl, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# -----------------------------
# TRAINING CURVES
# -----------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()
