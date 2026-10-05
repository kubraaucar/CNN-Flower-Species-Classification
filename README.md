#  CNN ile Çiçek Türü Sınıflandırma
<img width="1578" height="851" alt="Ekran görüntüsü 2026-10-05 215855" src="https://github.com/user-attachments/assets/5212264c-0009-4a94-aa65-e64ce63bccab" />


Bu proje, çiçek görsellerini sınıflandırmak amacıyla **TensorFlow ve Keras** kullanılarak geliştirilmiş bir Evrişimsel Sinir Ağı (Convolutional Neural Network - CNN) uygulamasıdır.

Model, **TensorFlow Flowers (`tf_flowers`)** veri seti üzerinde eğitilmiş ve verilen bir çiçek fotoğrafını 5 farklı çiçek türünden biri olarak sınıflandıracak şekilde geliştirilmiştir.

Ayrıca eğitilen model, **Gradio** ile hazırlanan web arayüzü üzerinden kullanıcı tarafından yüklenen fotoğraflar üzerinde tahmin yapabilmektedir.

---

##  Sınıflar

Model aşağıdaki 5 çiçek türünü sınıflandırmaktadır:

- Karahindiba (Dandelion)
- Papatya (Daisy)
- Lale (Tulips)
- Ayçiçeği (Sunflowers)
- Gül (Roses)

---

##  Veri Seti

Projede TensorFlow Datasets üzerinden sağlanan **`tf_flowers`** veri seti kullanılmıştır.

Veri seti eğitim, doğrulama ve test olmak üzere üç bölüme ayrılmıştır:

| Veri | Oran |
|---|---:|
| Eğitim (Train) | %70 |
| Doğrulama (Validation) | %15 |
| Test | %15 |

Test veri setinde toplam **550 görüntü** bulunmaktadır.

---

##  CNN Model Mimarisi

Projede özel bir CNN mimarisi oluşturulmuştur.

Model yapısı:

```text
Girdi Görüntüsü (180x180x3)
        ↓
Conv2D (32 filtre, ReLU)
        ↓
MaxPooling2D
        ↓
Conv2D (64 filtre, ReLU)
        ↓
MaxPooling2D
        ↓
Conv2D (128 filtre, ReLU)
        ↓
MaxPooling2D
        ↓
Flatten
        ↓
Dense (128, ReLU)
        ↓
Dropout (0.5)
        ↓
Dense (5, Softmax)
```

Model yaklaşık **6.65 milyon parametreye** sahiptir.

---

##  Veri Artırma (Data Augmentation)

Modelin yalnızca eğitim görüntülerini ezberlemesini azaltmak ve farklı görüntülere karşı daha iyi genelleme yapabilmesini sağlamak amacıyla veri artırma teknikleri uygulanmıştır.

Eğitim verilerinde kullanılan işlemler:

- Rastgele yatay çevirme
- Parlaklık değiştirme
- Kontrast değiştirme
- Rastgele kırpma
- Görüntü boyutlandırma
- Piksel değerlerini normalize etme

Doğrulama ve test görüntülerinde ise yalnızca yeniden boyutlandırma ve normalizasyon işlemleri uygulanmıştır.

---

##  Model Eğitimi

Model aşağıdaki ayarlar kullanılarak eğitilmiştir:

- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Loss Function:** Sparse Categorical Crossentropy
- **Batch Size:** 32
- **Maksimum Epoch:** 10

Eğitim sırasında aşağıdaki callback yapıları kullanılmıştır:

- **EarlyStopping:** Modelin gelişimi durduğunda gereksiz eğitimi önlemek için
- **ReduceLROnPlateau:** Doğrulama kaybındaki gelişim durduğunda öğrenme oranını azaltmak için
- **ModelCheckpoint:** En iyi modeli kaydetmek için

En iyi model:

```text
best_model.keras
```

dosyasında saklanmaktadır.

---

##  Model Performansı

Eğitilen CNN modeli test veri setinde:

**Test Accuracy: %73.45**

**Test Loss: 0.7170**

sonuçlarını elde etmiştir.

Sınıf bazında elde edilen sonuçlar:

| Sınıf | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Karahindiba | 0.84 | 0.72 | 0.78 |
| Papatya | 0.71 | 0.78 | 0.74 |
| Lale | 0.70 | 0.70 | 0.70 |
| Ayçiçeği | 0.74 | 0.90 | 0.81 |
| Gül | 0.69 | 0.59 | 0.64 |

En yüksek F1-score değeri **0.81 ile ayçiçeği sınıfında** elde edilmiştir.

Sonuçlar incelendiğinde modelin bazı çiçek türlerini diğerlerinden daha başarılı şekilde ayırt edebildiği görülmektedir.

---

##  Tahmin (Inference)

Eğitilmiş model, eğitim sürecinden bağımsız olarak yeni çiçek görüntüleri üzerinde tahmin yapabilmektedir.

Tahmin süreci:

```text
Kullanıcı Görseli
       ↓
180x180 Boyutlandırma
       ↓
Normalizasyon
       ↓
CNN Modeli
       ↓
Softmax Olasılıkları
       ↓
Çiçek Türü Tahmini
```

Örnek bir görüntü için model çıktısı:

```text
roses:       %55.69
tulips:      %20.48
daisy:       %19.56
sunflowers:   %2.99
dandelion:    %1.28
```

Bu sayede yalnızca en yüksek tahmin değil, modelin diğer sınıflara verdiği olasılıklar da görüntülenebilmektedir.

---

##  Gradio Web Arayüzü

Modelin daha kolay kullanılabilmesi için **Gradio** kullanılarak basit bir web arayüzü geliştirilmiştir.

Kullanıcı bir çiçek fotoğrafı yüklediğinde sistem:

```text
Fotoğraf Yükleme
       ↓
Görüntü Ön İşleme
       ↓
CNN Modeli
       ↓
Sınıf Olasılıkları
       ↓
Tahmin Sonuçları
```

adımlarını gerçekleştirerek en olası çiçek türlerini kullanıcıya göstermektedir.

---

##  Uygulama Görüntüsü

Gradio arayüzünün ekran görüntüsü bu bölüme eklenecektir.

<img width="1582" height="842" alt="Ekran görüntüsü 2026-10-05 215839" src="https://github.com/user-attachments/assets/94c15b03-431b-4e83-85a7-ca1b3b875337" />
<img width="1585" height="848" alt="Ekran görüntüsü 2026-10-05 215806" src="https://github.com/user-attachments/assets/5fa4414f-7d5e-4d03-bd0c-cc47186ec883" />
<img width="1577" height="852" alt="Ekran görüntüsü 2026-10-05 215723" src="https://github.com/user-attachments/assets/d897b46e-4a95-4041-8f22-3079f9e1ea90" />

---

##  Kurulum

Projeyi bilgisayarınıza klonlayın:

```bash
git clone https://github.com/kubraaucar/CNN-Flower-Species-Classification.git
```

Proje klasörüne girin:

```bash
cd CNN-Flower-Species-Classification
```

Sanal ortam oluşturun:

```bash
python -m venv venv
```

Windows üzerinde sanal ortamı aktif edin:

```bash
venv\Scripts\activate
```

Gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

---

##  Uygulamayı Çalıştırma

Gradio uygulamasını başlatmak için:

```bash
python app.py
```

Komut çalıştırıldıktan sonra terminalde gösterilen yerel Gradio adresini tarayıcıda açın.

Ardından bir çiçek fotoğrafı yükleyerek modelin tahminlerini görüntüleyebilirsiniz.

---

##  Proje Yapısı

```text
CNN-Flower-Species-Classification/
│
├── app.py
├── cnn.py
├── inference.py
├── best_model.keras
├── requirements.txt
├── .gitignore
└── README.md
```

### `cnn.py`

Veri setinin yüklenmesi, ön işleme, veri artırma, CNN modelinin oluşturulması, eğitilmesi ve değerlendirilmesi işlemlerini içerir.

### `inference.py`

Kaydedilmiş modeli yükler ve yeni görüntüler üzerinde tahmin yapılmasını sağlar.

### `app.py`

Gradio kullanılarak oluşturulan kullanıcı arayüzünü içerir.

### `best_model.keras`

Eğitim sırasında elde edilen en iyi modelin kaydedilmiş halidir.

---

##  Kullanılan Teknolojiler

- Python
- TensorFlow
- Keras
- TensorFlow Datasets
- NumPy
- Matplotlib
- Scikit-learn
- Gradio

---

##  Gelecekte Yapılabilecek Geliştirmeler

Projenin sonraki aşamalarında:

- MobileNetV2 veya EfficientNet ile Transfer Learning
- Flatten yerine Global Average Pooling kullanımı
- Hiperparametre optimizasyonu
- Veri artırma yöntemlerinin geliştirilmesi
- Yanlış sınıflandırılan görüntülerin analizi
- Model doğruluğunun artırılması
- Gradio uygulamasının çevrim içi yayınlanması

gibi geliştirmeler yapılabilir.

---
