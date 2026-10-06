import tensorflow as tf

# Çiçek sınıfları (tf_flowers etiket sırası)
CLASS_NAMES = [
    "dandelion",
    "daisy",
    "tulips",
    "sunflowers",
    "roses"
]

# Her model için dosya yolu, girdi boyutu ve 0-1 normalizasyonu gerekip gerekmediği.
# MobileNetV2 modeli ham 0-255 piksel alır, ölçekleme modelin içinde yapılır.
MODEL_CONFIGS = {
    "MobileNetV2 (Transfer Learning)": {
        "path": "mobilenetv2_model.keras",
        "img_size": (224, 224),
        "normalize": False,
    },
    "Custom CNN": {
        "path": "best_model.keras",
        "img_size": (180, 180),
        "normalize": True,
    },
}

DEFAULT_MODEL = "MobileNetV2 (Transfer Learning)"

# Modelleri ilk kullanımda yükle ve sakla
_loaded_models = {}


def get_model(model_name):
    if model_name not in _loaded_models:
        # sadece tahmin yapılacağı için derlemeye (optimizer, loss) gerek yok
        _loaded_models[model_name] = tf.keras.models.load_model(
            MODEL_CONFIGS[model_name]["path"], compile=False
        )
    return _loaded_models[model_name]


def preprocess_image(image, img_size, normalize):
    """
    Tahmin edilecek görüntüyü modelin beklediği
    formata dönüştürür.
    """

    # Görüntüyü TensorFlow tensorüne dönüştür
    image = tf.convert_to_tensor(image)

    # Modelin beklediği boyuta getir
    image = tf.image.resize(image, img_size)

    image = tf.cast(image, tf.float32)

    # Custom CNN 0-1 aralığında piksel bekler
    if normalize:
        image = image / 255.0

    # Model batch formatı beklediği için
    # (H, W, 3) -> (1, H, W, 3)
    image = tf.expand_dims(image, axis=0)

    return image


def predict_image(image, model_name=DEFAULT_MODEL):
    """
    Görüntüyü sınıflandırır ve
    sınıfların olasılıklarını döndürür.
    """

    config = MODEL_CONFIGS[model_name]

    image = preprocess_image(image, config["img_size"], config["normalize"])

    predictions = get_model(model_name).predict(image, verbose=0)

    probabilities = predictions[0]

    results = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }

    return results
