import tensorflow as tf
import numpy as np

# Modelin beklediği görüntü boyutu
IMG_SIZE = (180, 180)

# Çiçek sınıfları
CLASS_NAMES = [
    "dandelion",
    "daisy",
    "tulips",
    "sunflowers",
    "roses"
]

# Eğitilmiş modeli yükle
model = tf.keras.models.load_model("best_model.keras")


def preprocess_image(image):
    """
    Tahmin edilecek görüntüyü modelin beklediği
    formata dönüştürür.
    """

    # Görüntüyü TensorFlow tensorüne dönüştür
    image = tf.convert_to_tensor(image)

    # 180x180 boyutuna getir
    image = tf.image.resize(image, IMG_SIZE)

    # 0-255 değerlerini 0-1 arasına getir
    image = tf.cast(image, tf.float32) / 255.0

    # Model batch formatı beklediği için
    # (180, 180, 3) -> (1, 180, 180, 3)
    image = tf.expand_dims(image, axis=0)

    return image


def predict_image(image):
    """
    Görüntüyü sınıflandırır ve
    sınıfların olasılıklarını döndürür.
    """

    image = preprocess_image(image)

    predictions = model.predict(image, verbose=0)

    probabilities = predictions[0]

    results = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }

    return results

