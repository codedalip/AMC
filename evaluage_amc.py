# import numpy as np
# import pickle
# import tensorflow as tf
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, classification_report
# from tensorflow.keras.utils import to_categorical


# data_path = "RML2016.10a_dict.dat"

# with open(data_path, 'rb') as f:
#     dataset = pickle.load(f, encoding='latin1')

# X = []
# Y = []
# SNR = []

# for (mod, snr), signals in dataset.items():
#     for signal in signals:
#         X.append(signal)
#         Y.append(mod)
#         SNR.append(snr)

# X = np.array(X)
# Y = np.array(Y)
# SNR = np.array(SNR)

# X = np.transpose(X, (0,2,1))
# X = X / np.max(np.abs(X), axis=(1,2), keepdims=True)

# mods = sorted(list(set(Y)))
# mod_to_int = {mod:i for i,mod in enumerate(mods)}
# Y_int = np.array([mod_to_int[m] for m in Y])
# num_classes = len(mods)
# Y_onehot = to_categorical(Y_int, num_classes)


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test, snr_train, snr_test = train_test_split(
#     X, Y_onehot, SNR,
#     test_size=0.2,
#     random_state=42,
#     stratify=Y_int
# )


# model = tf.keras.models.load_model("amc_model.h5")
# print("Model loaded successfully.")

# y_pred = model.predict(X_test)
# y_pred_classes = np.argmax(y_pred, axis=1)
# y_true_classes = np.argmax(y_test, axis=1)


# cm = confusion_matrix(y_true_classes, y_pred_classes)

# plt.figure(figsize=(12,10))
# sns.heatmap(
#     cm,
#     annot=True,
#     fmt='d',
#     cmap="Blues",
#     xticklabels=mods,
#     yticklabels=mods
# )
# plt.title("Confusion Matrix (Counts)")
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.tight_layout()
# plt.show()


# print("\nOverall Accuracy: {:.2f}%".format(
#     np.mean(y_pred_classes == y_true_classes) * 100
# ))

# print("\nClassification Report:\n")
# print(classification_report(y_true_classes, y_pred_classes, target_names=mods))


# class_acc = cm.diagonal() / cm.sum(axis=1)

# print("\nPer-Class Accuracy:")
# for i, mod in enumerate(mods):
#     print(f"{mod:10s}: {class_acc[i]*100:.2f}%")


import numpy as np
import pickle
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


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


# reshape for CNN
X = np.transpose(X, (0,2,1))


# normalize
X = X / np.max(np.abs(X), axis=(1,2), keepdims=True)


# label encoding
mods = sorted(list(set(Y)))
mod_to_int = {mod:i for i,mod in enumerate(mods)}
Y_int = np.array([mod_to_int[m] for m in Y])
num_classes = len(mods)
Y_onehot = to_categorical(Y_int, num_classes)


# train test split
X_train, X_test, y_train, y_test, snr_train, snr_test = train_test_split(
    X, Y_onehot, SNR,
    test_size=0.2,
    random_state=42,
    stratify=Y_int
)


# load trained model
model = tf.keras.models.load_model("amc_model.h5")
print("Model loaded successfully.")


# prediction
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)


# ===============================
# CONFUSION MATRIX
# ===============================

cm = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(12,10))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap="Blues",
    xticklabels=mods,
    yticklabels=mods
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()


# ===============================
# OVERALL ACCURACY
# ===============================

overall_acc = np.mean(y_pred_classes == y_true_classes)

print("\nOverall Accuracy: {:.2f}%".format(overall_acc * 100))


# ===============================
# CLASSIFICATION REPORT
# ===============================

print("\nClassification Report:\n")
print(classification_report(y_true_classes, y_pred_classes, target_names=mods))


# ===============================
# PER CLASS ACCURACY
# ===============================

class_acc = cm.diagonal() / cm.sum(axis=1)

print("\nPer-Class Accuracy:")
for i, mod in enumerate(mods):
    print(f"{mod:10s}: {class_acc[i]*100:.2f}%")


# ===============================
# ACCURACY VS SNR
# ===============================

snr_unique = np.sort(np.unique(snr_test))

acc_snr = []

for snr in snr_unique:

    indices = np.where(snr_test == snr)

    y_true_snr = y_true_classes[indices]
    y_pred_snr = y_pred_classes[indices]

    acc = np.mean(y_true_snr == y_pred_snr)

    acc_snr.append(acc)


print("\nAccuracy vs SNR:")
for snr, acc in zip(snr_unique, acc_snr):
    print(f"SNR {snr:>3} dB : {acc*100:.2f}%")


# ===============================
# PLOT ACCURACY VS SNR
# ===============================

plt.figure(figsize=(8,5))

plt.plot(snr_unique, acc_snr, marker='o')

plt.xlabel("Signal-to-Noise Ratio (dB)")
plt.ylabel("Classification Accuracy")
plt.title("Accuracy vs SNR")

plt.grid(True)

plt.show()