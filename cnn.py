"""
flowers dataset:
    rgb: 224x224

CNN ile siniflandirma modeli olusturma ve problemi çözme    
"""
# import libraries
from tensorflow_datasets import load #veri seti yükleme
from tensorflow.data import AUTOTUNE #VERİ SETİ OPTİMİZASYONU
from tensorflow.keras.models import Sequential, load_model 
from tensorflow.keras.layers import(
    Conv2D, #2D convolutional layer
    MaxPooling2D, # max pooling layer
    GlobalAveragePooling2D, # her feature map'in ortalamasını alarak tek boyutlu hale getirme
    BatchNormalization, # her batch'te aktivasyonları normalize eder, egitimi hizlandirir ve stabil hale getirir
    Activation, # aktivasyon fonksiyonunu ayri katman olarak uygulamak icin (Conv -> BN -> ReLU)
    Dense, # tam baglantili katman , karar verme(classification)
    Dropout # rastgele noronları kapatma ve overfitting engelleme, ezberi engeller
)
from tensorflow.keras.optimizers import Adam #optimizer
from tensorflow.keras.callbacks import(
    EarlyStopping, #erken durdurma
    ReduceLROnPlateau, #oğrenme oranını azaltma
    ModelCheckpoint # model kaydetme, en iyi modeli kaybetmemek
)

import json
import os

import tensorflow as tf
import matplotlib
matplotlib.use("Agg") #grafikleri pencere acmadan dosyaya kaydet
import matplotlib.pyplot as plt

import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# tekrarlanabilirlik: python, numpy ve tensorflow icin sabit seed
SEED = 42
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism() #ayni seed ile ayni sonuclari almak icin

IMAGES_DIR = "images" #grafiklerin kaydedilecegi klasor
RUN_NAME = "cnn_deep" #grafik ve metrik dosyalarinin on eki (4 bloklu, BatchNorm'lu, GAP'li CNN)
os.makedirs(IMAGES_DIR, exist_ok=True)

# veri seti yükleme

(ds_train, ds_val, ds_test), ds_info = load(
    "tf_flowers",
    split=[
        "train[:70%]",      # %70 eğitim
        "train[70%:85%]",   # %15 validation
        "train[85%:]"       # %15 test
    ],
    as_supervised=True,
    with_info=True
)

print(ds_info.features) #veri seti hakkında bilgi yazdırma
print("number og classes:", ds_info.features['label'].num_classes)


# örnek veri görsellestirme
#egitim setinden rastgele 3 tane resim ve etiket alalım
fig = plt.figure(figsize= (10,5))
for i, (image, label) in enumerate(ds_train.take(3)):
    ax = fig.add_subplot(1, 3, i+1) #1 satır, 3 sütun, i+1. resim
    ax.imshow(image.numpy().astype("uint8")) #resmi gorsellestirme
    ax.set_title(f"Etiket: {label.numpy()}") #etiket baslik olarak yazdırma
    ax.axis("off") #eksenleri kapatma

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "sample_images.png"), dpi=120) #grafiği kaydetme
plt.close()


IMG_SIZE = (180, 180)

# data augmentation + preprocessing
def preprocess_train(image, label):
    """
    resize, random flip, brightness, contrast, crop
    normalize
    """
    image = tf.image.resize(image, IMG_SIZE) #boyutlandırma
    image = tf.image.random_flip_left_right(image) #yatay olarak rastgele cevirme
    image = tf.image.random_brightness(image, max_delta=0.1) #rastgele parlaklık
    image = tf.image.random_contrast(image, lower=0.9, upper = 1.2)#rastgele kontrast
    image = tf.image.random_crop(image, size=(160, 160, 3)) #rastgele crop
    image = tf.image.resize(image, IMG_SIZE) #tekrar boyutlandırma
    image = tf.cast(image, tf.float32)/255.0 #normalize etme
    return image, label 

def preprocess_val(image, label):
    """
    resize, normalize
    """
    image = tf.image.resize(image, IMG_SIZE) #boyutlandırma
    image = tf.cast(image, tf.float32)/255.0 #normalize etme
    return image, label

#veri seti hazırlama
ds_train = (
    ds_train
    .map(preprocess_train, num_parallel_calls=AUTOTUNE) #on isleme ve augmentasyon
    .shuffle(1000, seed=SEED) #karıstırma
    .batch(32) #batch boyutu
    .prefetch(AUTOTUNE) #veri setini önceden hazırlamak
)


ds_val = (
    ds_val
    .map(preprocess_val, num_parallel_calls=AUTOTUNE) #on isleme
    .batch(32) #batch boyutu
    .prefetch(AUTOTUNE) #veri setini önceden hazırlamak
)

ds_test = (
    ds_test
    .map(preprocess_val, num_parallel_calls=AUTOTUNE)
    .batch(32)
    .prefetch(AUTOTUNE)
)

# CNN modelini olusturma

model = Sequential([

    #Feature Extraction Layers: her blok Conv -> BatchNorm -> ReLU -> MaxPooling
    Conv2D(32, (3,3), use_bias = False, input_shape = (*IMG_SIZE, 3)), #32filtre sayısı, 3x3kernel, 3kanal(RGB); bias'ı BatchNorm üstlenir
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D((2,2)),  #2x2 max pooling

    Conv2D(64, (3,3), use_bias = False), #64 filtre,3x3 kernel
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D((2,2)), #2x2 max pooling

    Conv2D(128, (3,3), use_bias = False), #128 filtre,3x3 kernel
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D((2,2)), #2x2 max pooling

    #4. blok: daha derin ozellikler ve daha genis gorus alani (GAP'li 3 blokluk model underfit oluyordu)
    Conv2D(256, (3,3), use_bias = False), #256 filtre,3x3 kernel
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D((2,2)), #2x2 max pooling

    #Classification Layers
    GlobalAveragePooling2D(), #Flatten yerine: parametre sayısını büyük ölçüde azaltır, overfitting'i azaltır
    Dense(128, activation = "relu"),
    Dropout(0.5), #overfitting i engellemek için dropout
    Dense(ds_info.features["label"].num_classes, activation = "softmax")  #cikis katmanı, softmax aktivasyonu

])

# callback
callbacks = [
    #eger val loss 5 epoch boyunca iyilesmezse egitimi durdur ve en iyi agırlıkları yukle
    EarlyStopping(monitor = "val_loss", patience = 5, restore_best_weights = True),

    #val loss 3 epoch boyunca iyilesmezse learning rate 0.2 çarpanı ile azalt
    ReduceLROnPlateau(monitor = "val_loss", factor = 0.2, patience = 3, verbose = 1, min_lr = 1e-9), #ogrenme oranını azaltma

    #her epoch sonunda eger model daha iyiise kaybolur
    ModelCheckpoint(
        "best_model.keras",
        monitor="val_loss",
        save_best_only=True
) #model kaydetme , en iyi modeli kaydet
]


# derleme 
model.compile(
    optimizer = Adam(learning_rate = 0.001), #adam optimizer, ogrenme oranını 0.001 olarak ayarla
    loss = "sparse_categorical_crossentropy", #kayıp fonksiyonu, etiketler tamsayı olduğu icin sparse kullan
    metrics = ["accuracy"] #metrik olarak dogruluk kullan
)

print(model.summary()) #model ozeti

# traning
history = model.fit(
    ds_train, #egitim veri seti
    validation_data = ds_val, #validasyon veri seti
    epochs = 40, #maksimum epoch sayısı, EarlyStopping daha önce durdurabilir
    callbacks = callbacks, 
    verbose = 1 #egitim ilerlemesini göster
)
# Validation loss'a göre kaydedilen en iyi modeli yükle
model = load_model("best_model.keras")

# Model seçimi validation setine göre yapıldı; seçilen modelin validation sonucu
val_loss, val_accuracy = model.evaluate(ds_val)

print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Test veri seti üzerinde modeli değerlendir (sadece raporlama için)
test_loss, test_accuracy = model.evaluate(ds_test)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Gerçek etiketler ve model tahminleri
y_true = []
y_pred = []

for images, labels in ds_test:

    predictions = model.predict(images, verbose=0)

    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)


# Sınıf isimleri
class_names = ds_info.features["label"].names

print("Sınıflar:", class_names)


# Precision, Recall ve F1-score
print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# sonuçları dosyaya kaydet (README ve karşılaştırma için)
with open(os.path.join(IMAGES_DIR, f"{RUN_NAME}_metrics.json"), "w") as f:
    json.dump({
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "report": classification_report(y_true, y_pred, target_names=class_names, output_dict=True),
        "confusion_matrix": cm.tolist(),
    }, f, indent=2)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(cmap="Blues")
plt.title("Custom CNN - Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, f"{RUN_NAME}_confusion_matrix.png"), dpi=120)
plt.close()


# model evaluation
plt.figure(figsize=(12,5))

#dogruluk grafigi
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label = "Train")
plt.plot(history.history["val_accuracy"], label = "Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Custom CNN - Accuracy")
plt.legend()

#loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label = "Train")
plt.plot(history.history["val_loss"], label = "Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Custom CNN - Loss")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, f"{RUN_NAME}_training_curves.png"), dpi=120)  #grafigi kaydet
plt.close()

