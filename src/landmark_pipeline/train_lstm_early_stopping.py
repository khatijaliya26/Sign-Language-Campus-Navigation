import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# -----------------------------
# CONFIGURATION
# -----------------------------
SEGMENT_ROOT = os.path.join("segmented_landmarks")
MAX_FRAMES = 30
NUM_FEATURES = 63
TEST_SIZE = 0.2
RANDOM_STATE = 42

# -----------------------------
# LOAD DATA
# -----------------------------
X = []
y = []

classes = sorted(os.listdir(SEGMENT_ROOT))
print("Classes:", classes)

for class_name in classes:
    class_path = os.path.join(SEGMENT_ROOT, class_name)
    for file in os.listdir(class_path):
        if file.endswith(".npy"):
            data = np.load(os.path.join(class_path, file))
            X.append(data)
            y.append(class_name)

X = np.array(X)
y = np.array(y)

print("Data shape:", X.shape)

# -----------------------------
# LABEL ENCODING
# -----------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_onehot,
    test_size=TEST_SIZE,
    stratify=y_encoded,
    random_state=RANDOM_STATE
)

# -----------------------------
# MODEL DEFINITION
# -----------------------------
model = Sequential([
    Input(shape=(MAX_FRAMES, NUM_FEATURES)),
    Bidirectional(LSTM(64)),
    Dropout(0.4),
    Dense(64, activation="relu"),
    Dense(len(classes), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# EARLY STOPPING
# -----------------------------
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=100,                 # large number, early stopping will handle it
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# -----------------------------
# EVALUATION
# -----------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("\nTest Accuracy:", round(test_acc, 4))

y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(
    classification_report(
        y_true_labels,
        y_pred_labels,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_true_labels, y_pred_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Early Stopping)")
plt.tight_layout()
plt.show()

# -----------------------------
# TRAINING CURVES
# -----------------------------
plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve (Early Stopping)")
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve (Early Stopping)")
plt.legend()
plt.show()

# -----------------------------
# SAVE MODEL
# -----------------------------
os.makedirs("models", exist_ok=True)
model.save("models/lstm_early_stopping_model.h5")

print("\nModel saved to models/lstm_early_stopping_model.h5")
