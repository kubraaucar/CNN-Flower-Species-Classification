"""
flowers dataset:
    rgb: 224x224

CNN ile siniflandirma modeli olusturma ve problemi çözme    
"""
# import libraries
from tensorflow_datasets import load #veri seti yükleme
from tensorflow.data import AUTOTUNE #VERİ SETİ OPTİMİZASYONU
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import(
    Conv2D, #2D convolutional layer 
    MaxPooling2D, # max pooling layer 
    Flatten, # çok botutlu veriyi tek boyutlu hale getirme
    Dense, # tam baglantili katman , karar verme(classification)
    Dropout # rastgele noronları kapatma ve overfitting engelleme, ezberi engeller
)
from tensorflow.keras.optimizers import Adam #optimizer
from tensorflow.keras.callbacks import(
    EarlyStopping, #erken durdurma
    ReduceLROnPlateau, #oğrenme oranını azaltma
    ModelCheckpoint # model kaydetme, en iyi modeli kaybetmemek
)

import tensorflow as tf
import matplotlib.pyplot as plt 

# veri seti yükleme

(ds_train, ds_val), ds_info = load(
    "tf_flowers", #veri seti ismi
    split = ["train[:80%]", #veri setinin %80 i eğitim için
           "train[80%:]"], #veri setinin %20 si test için

    as_supervised=True, #veri setinin görsel etiket çiftinin olması
    with_info=True # veri seti hakkında bilgi alma
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
plt.show() #grafiği gosterme 


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
    .shuffle(1000) #karsılastırma 
    .batch(32) #batch boyutu
    .prefetch(AUTOTUNE) #veri setini önceden hazırlamak
)


ds_val = (
    ds_val
    .map(preprocess_val, num_parallel_calls=AUTOTUNE) #on isleme
    .batch(32) #batch boyutu
    .prefetch(AUTOTUNE) #veri setini önceden hazırlamak
)

# CNN modelini olusturma

model = Sequential([

    #Feature Extraction Layers
    Conv2D(32, (3,3), activation = "relu", input_shape = (*IMG_SIZE, 3)), #32filtre sayısı, 3x3kernel, relu aktivasyon fonks.,3kanal(RGB)
    MaxPooling2D((2,2)),  #2x2 max pooling

    Conv2D(64, (3,3), activation = "relu"), #64 filtre,3x3 kernel, relu aktivasyon
    MaxPooling2D((2,2)), #2x2 max pooling

    Conv2D(128, (3,3), activation = "relu"), #128 filtre,3x3 kernel, relu aktivasyon
    MaxPooling2D((2,2)), #2x2 max pooling

    #Classification Layers
    Flatten(), #çok boyutlu veriyi vektöre çevirme
    Dense(128, activation = "relu"),
    Dropout(0.5), #overfitting i engellemek için dropout
    Dense(ds_info.features["label"].num_classes, activation = "softmax")  #cikis katmanı, softmax aktivasyonu

])

# callback
callbacks = [
    #eger val loss 3 epoch bpyunca iyilesmezse egitimi durdur ve en iyi agırlıkları yukle
    EarlyStopping(monitor = "val_loss", patience = 3, restore_best_weights = True),

    #val loss 2 epoch boyunca iyilesmezse learning rate 0.2 çarpanı ile azalt
    ReduceLROnPlateau(monitor = "val_loss", factor = 0.2, patience = 2, verbose = 1, min_lr = 1e-9), #ogrenme oranını azaltma

    #her epoch sonunda eger model daha iyiise kaybolur
    ModelCheckpoint("best_model.h5", save_best_only=True) #model kaydetme , en iyi modeli kaydet
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
    epochs = 10, #epoch sayısı
    callbacks = callbacks, 
    verbose = 1 #egitim ilerlemesini göster
)


# model evaluation
plt.figure(figsize=(12,5))

#dogruluk grafigi
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label = "Egitim Dogrulugu")
plt.plot(history.history["val_accuracy"], label = "Validasyon Dogrulugu")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.legend()

#loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label = "Egitim Kaybi")
plt.plot(history.history["val_loss"], label = "Validasyon KAybi")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model Loss")
plt.legend()

plt.tight_layout()
plt.show()  #grafigi goster

