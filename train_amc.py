import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical

print("Devices:", tf.config.list_physical_devices())


data_path = "RML2016.10a_dict.dat"

with open(data_path, 'rb') as f:
    dataset = pickle.load(f, encoding='latin1')

X = []
Y = []
SNR = []

for (mod, snr), signals in dataset.items():
    for signal in signals:
        X.append(signal)
        Y.append(mod)
        SNR.append(snr)

X = np.array(X)
Y = np.array(Y)
SNR = np.array(SNR)

print("Original shape:", X.shape)  # (220000, 2, 128)


X = np.transpose(X, (0, 2, 1))

X = X / np.max(np.abs(X), axis=(1,2), keepdims=True)

mods = sorted(list(set(Y)))
mod_to_int = {mod:i for i,mod in enumerate(mods)}
Y_int = np.array([mod_to_int[m] for m in Y])

num_classes = len(mods)
Y_onehot = to_categorical(Y_int, num_classes)

X_train, X_test, y_train, y_test, snr_train, snr_test = train_test_split(
    X, Y_onehot, SNR,
    test_size=0.2,
    random_state=42,
    stratify=Y_int
)

print("Train shape:", X_train.shape)
print("Num classes:", num_classes)


def residual_block(x, filters, kernel_size=3):
    shortcut = x

    x = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv1D(filters, 1, padding='same')(shortcut)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)

    return x


def build_model(input_shape=(128,2), num_classes=11):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv1D(64, 3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = residual_block(x, 64)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = residual_block(x, 128)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = residual_block(x, 256)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)


model = build_model(input_shape=(128,2), num_classes=num_classes)
model.summary()


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1
)


history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=35,
    batch_size=512,
    callbacks=[lr_scheduler]
)


y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(10,8))
sns.heatmap(cm, xticklabels=mods, yticklabels=mods, cmap="Blues")
plt.title("Overall Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print(classification_report(y_true_classes, y_pred_classes, target_names=mods))


model.save("amc_model.h5")
print("Model saved successfully.")


snr_unique = np.unique(snr_test)
acc_snr = []

for snr in snr_unique:
    idx = np.where(snr_test == snr)
    X_s = X_test[idx]
    y_s = y_test[idx]

    y_pred_s = model.predict(X_s, verbose=0)
    acc = np.mean(np.argmax(y_pred_s, axis=1) ==
                  np.argmax(y_s, axis=1))
    acc_snr.append(acc)

plt.figure(figsize=(8,5))
plt.plot(snr_unique, acc_snr, marker='o')
plt.xlabel("SNR (dB)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs SNR")
plt.grid()
plt.show()




import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))

# -------- Accuracy Plot --------
plt.subplot(1,2,1)

plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)


# -------- Loss Plot --------
plt.subplot(1,2,2)

plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)


plt.tight_layout()

# Save high resolution for paper
plt.savefig("training_curves.png", dpi=600, bbox_inches="tight")

plt.show()