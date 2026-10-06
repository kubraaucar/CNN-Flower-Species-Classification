"""
tf_flowers veri seti üzerinde MobileNetV2 ile transfer learning.

cnn.py ile aynı veri bölmesi (%70 / %15 / %15) kullanılır, böylece
iki model aynı 550 test görüntüsü üzerinde karşılaştırılabilir.

Eğitim iki aşamalıdır:
    1) Feature extraction: ImageNet ağırlıklı gövde dondurulur,
       sadece yeni sınıflandırma katmanı eğitilir.
    2) Fine-tuning: gövdenin son katmanları çözülür ve çok küçük
       bir öğrenme oranı ile birlikte eğitilir.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")  # grafikleri pencere açmadan dosyaya kaydet
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow_datasets import load
from tensorflow.data import AUTOTUNE
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_PATH = "mobilenetv2_model.keras"
IMAGES_DIR = "images"
FINE_TUNE_LAYERS = 30  # gövdenin çözülecek son katman sayısı

tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()  # cnn.py ile aynı şekilde tekrarlanabilir
os.makedirs(IMAGES_DIR, exist_ok=True)

# veri seti yükleme (cnn.py ile aynı bölme)
(ds_train, ds_val, ds_test), ds_info = load(
    "tf_flowers",
    split=["train[:70%]", "train[70%:85%]", "train[85%:]"],
    as_supervised=True,
    with_info=True
)
class_names = ds_info.features["label"].names
num_classes = len(class_names)


def resize(image, label):
    # piksel değerleri 0-255 aralığında kalır, normalizasyon modelin içinde yapılır
    return tf.image.resize(image, IMG_SIZE), label


ds_train = ds_train.map(resize, num_parallel_calls=AUTOTUNE).shuffle(1000, seed=42).batch(BATCH_SIZE).prefetch(AUTOTUNE)
ds_val = ds_val.map(resize, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
ds_test = ds_test.map(resize, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

# veri artırma katmanları sadece eğitim sırasında aktiftir
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.15),
], name="data_augmentation")

base_model = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
base_model.trainable = False

# model ham (0-255) görüntü alır; MobileNetV2'nin beklediği [-1, 1] ölçeklemesi modelin içindedir
inputs = layers.Input(shape=(*IMG_SIZE, 3))
x = data_augmentation(inputs)
x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)
x = base_model(x, training=False)  # BatchNorm katmanları fine-tuning sırasında da inference modunda kalır
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = Model(inputs, outputs)

# 1. aşama: feature extraction
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

checkpoint = ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True)
history_1 = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=10,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, verbose=1),
        checkpoint,
    ]
)

# 2. aşama: fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_2 = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=20,  # EarlyStopping daha önce durdurabilir
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        # sadece 1. aşamanın en iyi val_loss değerinden daha iyiyse kaydet
        ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True,
                        initial_value_threshold=checkpoint.best),
    ]
)

# iki aşamanın validation loss'a göre en iyisi
model = load_model(MODEL_PATH)
val_loss, val_accuracy = model.evaluate(ds_val)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# test değerlendirmesi (sadece raporlama için)
test_loss, test_accuracy = model.evaluate(ds_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

y_true, y_pred = [], []
for images, labels in ds_test:
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(model.predict(images, verbose=0), axis=1))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=2))

report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
cm = confusion_matrix(y_true, y_pred)
with open(os.path.join(IMAGES_DIR, "mobilenetv2_metrics.json"), "w") as f:
    json.dump({"val_loss": val_loss, "val_accuracy": val_accuracy,
               "test_loss": test_loss, "test_accuracy": test_accuracy, "report": report,
               "confusion_matrix": cm.tolist()}, f, indent=2)

# confusion matrix
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot(cmap="Blues")
plt.title("MobileNetV2 - Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "mobilenetv2_confusion_matrix.png"), dpi=120)
plt.close()

# eğitim grafikleri (iki aşama arka arkaya, dikey çizgi fine-tuning başlangıcı)
acc = history_1.history["accuracy"] + history_2.history["accuracy"]
val_acc = history_1.history["val_accuracy"] + history_2.history["val_accuracy"]
loss = history_1.history["loss"] + history_2.history["loss"]
val_loss = history_1.history["val_loss"] + history_2.history["val_loss"]
fine_tune_start = len(history_1.history["loss"])

plt.figure(figsize=(12, 5))
for i, (train_values, val_values, title) in enumerate(
        [(acc, val_acc, "Accuracy"), (loss, val_loss, "Loss")]):
    plt.subplot(1, 2, i + 1)
    plt.plot(train_values, label="Train")
    plt.plot(val_values, label="Validation")
    plt.axvline(fine_tune_start - 0.5, color="gray", linestyle="--", label="Fine-tuning start")
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.title(f"MobileNetV2 - {title}")
    plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "mobilenetv2_training_curves.png"), dpi=120)
plt.close()
