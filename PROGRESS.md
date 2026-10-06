# İlerleme Notları

Branch: `improvements` · Son güncelleme: 2026-10-06

Veri bölmesi (`tf_flowers`): eğitim `train[:70%]` (2569), validation `train[70%:85%]` (551), test `train[85%:]` (550 görüntü).
**Model seçimi validation loss'a göre yapılıyor, test seti sadece en sonda raporlanıyor.**
Ham metrikler ve confusion matrix'ler `images/*_metrics.json` dosyalarında.

## Sonuçlar

| Model | Dosya | Parametre | Val Acc | Val Loss | Test Acc | Test Loss |
|---|---|---:|---:|---:|---:|---:|
| Baseline CNN (3 blok, Flatten) | `baseline_model.keras` (git dışı) | 6.65M | %72.78 | 0.766 | %73.45 | 0.717 |
| CNN + GAP (3 blok) | `cnn_gap_model.keras` | 110K | %68.06 | 0.810 | %70.55 | 0.753 |
| Derin CNN (4 blok, BatchNorm, GAP) | `best_model.keras` | 423K | %80.76 | 0.533 | %83.09 | 0.482 |
| MobileNetV2, sadece 1. aşama (gövde dondurulmuş) | `mobilenetv2_frozen.keras` (git dışı) | 2.26M | %89.47 | 0.302 | %89.64 | 0.278 |
| **MobileNetV2 + fine-tuning (BN donuk)** | `mobilenetv2_model.keras` | 2.26M | **%91.65** | **0.244** | **%93.45** | **0.195** |

### Sınıf bazlı F1 (precision / recall)

| Sınıf | Baseline | CNN + GAP | Derin CNN | MobileNetV2 + FT |
|---|---|---|---|---|
| Karahindiba | 0.78 (0.84 / 0.72) | 0.71 (0.70 / 0.72) | 0.86 (0.82 / 0.91) | 0.97 (0.96 / 0.98) |
| Papatya | 0.74 (0.71 / 0.78) | 0.75 (0.72 / 0.78) | 0.83 (0.86 / 0.80) | 0.94 (0.92 / 0.96) |
| Lale | 0.70 (0.70 / 0.70) | 0.70 (0.72 / 0.68) | 0.82 (0.78 / 0.87) | 0.92 (0.92 / 0.91) |
| Ayçiçeği | 0.81 (0.74 / 0.90) | 0.81 (0.76 / 0.88) | 0.90 (0.91 / 0.88) | 0.93 (0.97 / 0.90) |
| Gül | 0.64 (0.69 / 0.59) | 0.54 (0.60 / 0.49) | 0.74 (0.80 / 0.68) | 0.91 (0.90 / 0.93) |

### Karışıklık sayıları (test)

| | Baseline | CNN + GAP | Derin CNN | MobileNetV2 (1. aşama) | MobileNetV2 + FT |
|---|---:|---:|---:|---:|---:|
| Toplam hata | 146 | 162 | 93 | 57 | 36 |
| Gül → lale | 27 | 28 | 21 | 11 | 5 |
| Lale → gül | 19 | 26 | 11 | 7 | 10 |
| **Gül ↔ lale toplam (hataların payı)** | **46 (%32)** | **54 (%33)** | **32 (%34)** | **18 (%32)** | **15 (%42)** |
| Karahindiba → ayçiçeği | 16 | 19 | 4 | 3 | 0 |

## Yorumlar

- **Kendi CNN'imin gelişimi: %73.45 → %70.55 → %83.09.**
  - **GAP tek başına underfit oldu.** Eğitim accuracy'si yaklaşık %72, validation accuracy'si yaklaşık %69 kaldı. Baseline'ın kapasitesi büyük ölçüde Flatten'dan sonraki 6.4M parametrelik Dense katmanındaydı. Üç conv bloğundan sonra her nöronun gördüğü alan yaklaşık 22×22 piksel, bu yüzden model yerel renk ve doku ortalamasıyla karar veriyordu. Buna ek olarak ReduceLROnPlateau patience 2 olduğu için LR çok hızlı düştü.
  - **Derin CNN bu teşhisle tasarlandı.** 4. conv bloğu (256 filtre) görüş alanını genişletti. BatchNorm eğitimi hızlandırdı ve stabil hale getirdi. ReduceLROnPlateau patience'ı 3'e çıkarıldı. Model 32. epoch'ta durdu, en iyi epoch 27 oldu. Eğitim accuracy'si %83, validation accuracy'si %81, yani underfitting kalmadı. Sonuç baseline'dan 9.6 puan daha iyi ve parametre sayısı 16 kat daha az.
- **MobileNetV2 fine-tuning'i: BatchNorm'u donuk tutmak belirleyici oldu.**
  - **1. deneme (açılan katmanlardaki BN eğitilebilir, patience 3):** katmanlar açılınca eğitim loss'u 0.27'den 0.50'ye sıçradı ve validation loss kötüleşti. Fine-tuning 4 epoch sonra durdu ve 1. aşama modeli seçildi. Grafik: `images/mobilenetv2_ft_trainable_bn_training_curves.png`.
  - **2. deneme (açılan 30 katmandaki 11 BN donuk, sadece 10 conv katmanı eğitiliyor, patience 5):** sıçrama olmadı. Validation loss fine-tuning'in ilk epoch'unda 0.302'den 0.294'e indi ve 17. epoch'ta 0.244'e kadar düştü.
  - **Karar kuralı** (validation loss < 0.30177, yani 1. aşamanın tam değeri) test sonucuna bakılmadan uygulandı ve yeni model seçildi. Test sonucu %89.64'ten %93.45'e çıktı.
  - Determinism çalışıyor: 1. aşama iki çalıştırmada da validation loss'ta birebir aynı değerleri verdi.
  - Fine-tuning 20 epoch'un hepsini koştu. Son iyileşme 17. epoch'taydı ve EarlyStopping tetiklenmedi. Validation loss 0.25 civarında düzleşti, eğitim ile validation arasındaki fark açılıyor (%98'e karşı %92). Bu yüzden daha fazla epoch'tan büyük bir kazanç beklemiyorum.
- **Gül–lale en zor ayrım.** Mutlak sayı 46'dan 15'e indi, ama en iyi modelde hataların %42'si bu çiftten geliyor. Diğer hatalar daha hızlı azaldı, geriye zor ayrım kaldı. İki çiçeğin renk paleti (kırmızı, pembe, sarı) ve kupa biçimli taç yaprakları benzer. En iyi modelde yön de değişti: artık daha çok lale gül sanılıyor (10), tersi azaldı (5).
- **Karahindiba → ayçiçeği hatası 16'dan 0'a indi.** Sığ CNN'ler büyük ölçüde "sarı" rengine bakıyordu. Daha geniş görüş alanı ve ImageNet'te öğrenilen şekil özellikleri bu hatayı tamamen kaldırdı.

## Gradio testi (5. adım)

- Uygulama ayrı bir klasörde çalıştırıldı ve HTTP API üzerinden (`/predict_image`, gradio_client) test setinden sarı-turuncu bir gül fotoğrafıyla sorgulandı.
- **Ön işleme doğru.** Uygulamanın olasılıkları, değerlendirme pipeline'ının çıktısıyla iki modelde de birebir aynı (fark 0). Derin CNN'de 180px ve /255, MobileNetV2'de 224px ve ham piksel kullanılıyor.
- Aynı görüntüde modellerin tahminleri:

  | Model | Tahmin |
  |---|---|
  | Derin CNN | lale 0.60, gül 0.27 (yanlış) |
  | MobileNetV2, 1. aşama | lale 0.60, gül 0.37 (yanlış) |
  | MobileNetV2 + FT (varsayılan model) | **gül 0.82**, lale 0.18 (doğru, `inference.predict_image` ile) |

  Bu görüntü README'deki gül–lale yorumu için iyi bir örnek.

## Space hazırlığı (6. adım)

- **Optimizer state'i silindi.**
  - Keras 3'te `include_optimizer=False`, `.keras` formatında sessizce yok sayılıyor (sadece `.h5` için geçerli). Bu yüzden modeller `load_model(..., compile=False)` ile yüklenip yeniden kaydedildi.
  - `model.weights.h5` içinde artık `optimizer` grubu yok.
  - Boyutlar: derin CNN 5.2MB'tan 1.8MB'a, MobileNetV2 21.8MB'tan 9.7MB'a indi.
- **Tahminler değişmedi.** Validation ve test setindeki 1101 görüntüde olasılıklar bit düzeyinde aynı (max fark 0). Test accuracy'leri de aynı: %83.09 ve %93.45.
- **Kökteki model dosyaları da bu küçük sürümlerle değiştirildi.** `space/` içindekilerle byte olarak aynılar, bu yüzden Git ikisini tek blob olarak saklıyor. Not: bu dosyalardan eğitime kaldığı yerden devam edilemez. Yeniden eğitim script'leri zaten sıfırdan başlıyor.
- `inference.py` modelleri `compile=False` ile yüklüyor.
- `space/` klasörünün içeriği:
  - `app.py`, `inference.py` ve iki model: kökteki dosyaların kopyası. **Kökte değişiklik yapınca buraya da kopyalanmalı.**
  - `requirements.txt`: tensorflow 2.20.0, keras 3.13.2 ve numpy 2.4.2. Gradio, YAML'daki `sdk_version: 6.29.1` ile kuruluyor.
  - `README.md`: Spaces YAML başlığı ve İngilizce kısa açıklama.
- `space/` klasöründen yerelde import ve tahmin testi yapıldı: gül örneğinde MobileNetV2 gül 0.82, CNN lale 0.60.
- **Deploy denemesi (2026-10-06):**
  - Hesap `kuubraucar1` oldu (`hf auth whoami` çıktısı). `kubraucar1` diye bir hesap yok.
  - `create_repo(space_sdk="gradio")` çağrısı **402 Payment Required** döndürdü: "Static Spaces are free for everyone, but hosting Gradio and Docker Spaces on free cpu-basic requires a PRO subscription."
  - Space oluşturulmadı, Hub'da yarım kalan bir şey yok.

## Yapılan değişiklikler

- `cnn.py`:
  - Seed (42), `enable_op_determinism()` ve shuffle seed'i.
  - 4 conv bloğu (32, 64, 128 ve 256 filtre), her blok Conv → BatchNorm → ReLU → MaxPool sırasıyla.
  - `GlobalAveragePooling2D`, en fazla 40 epoch, EarlyStopping patience 5, ReduceLROnPlateau patience 3.
  - Validation ve test metrikleri ile grafikler `images/cnn_deep_*` dosyalarına kaydediliyor.
- `transfer_learning.py`:
  - MobileNetV2 (ImageNet, 224×224). Ölçekleme ve veri artırma modelin içinde.
  - 1. aşama: gövde dondurulmuş, lr 1e-3, 10 epoch.
  - 2. aşama: son 30 katman açılıyor ama içlerindeki BatchNorm'lar donuk, lr 1e-5, en fazla 20 epoch, EarlyStopping patience 5.
  - Seed, op determinism ve shuffle seed'i var. Checkpoint mesajları log'a yazılıyor.
- `inference.py` / `app.py`: iki model destekleniyor ve her modelin girdi boyutu ile normalizasyonu ayrı tanımlı. Varsayılan model MobileNetV2, validation'a göre en iyi model de o.
- Model dosyaları:
  - `baseline_model.keras` (80MB): git dışında.
  - `mobilenetv2_frozen.keras` (10MB): git dışında yerel yedek.
  - `cnn_gap_model.keras`: GAP modelinin yedeği.
  - `best_model.keras`: derin CNN.
  - `mobilenetv2_model.keras` (22MB, optimizer state'i içinde): fine-tuned model.
- Eğitim log'ları `logs/` klasöründe ve git dışında. `.claude/settings.local.json` git dışında ve Claude attribution'ı kapalı.

## Adımlar

- [x] 1. Hazırlık: branch açıldı, baseline yedeklendi, seed eklendi
- [x] 2. CNN iyileştirme: GAP (%70.55), ardından derin CNN (%83.09)
- [~] 3. Grafikler ve confusion matrix kaydediliyor, yeni grafiklerin etiketleri İngilizce. Kalan: gül–lale yorumunu README'ye yazmak
- [x] 4. MobileNetV2 transfer learning ve fine-tuning (%93.45)
- [x] 5. `inference.py` / `app.py` güncellendi ve iki modelle gerçek API üzerinden test edildi
- [x] 6. Model dosyaları optimizer'sız kaydedildi (bkz. "Space hazırlığı"), `space/` klasörü sade `requirements.txt` ile hazır
- [ ] 7. Hugging Face Spaces deploy: **engellendi**. Gradio Space'i ücretsiz `cpu-basic`'te barındırmak PRO abonelik gerektiriyor (bkz. açık karar 1)
- [ ] 8. README'yi İngilizce yeniden yazmak
- [ ] 9. GitHub About ve topic'ler, main'e merge

## Açık kararlar

1. **Canlı demo nerede çalışacak?**
   - A) Hugging Face PRO abonelik. `space/` klasörü hazır olduğu için yükleme tek komut.
   - B) Ücretsiz statik Space. Modeli TensorFlow.js veya ONNX'e çevirip tarayıcıda çalıştırmak gerekir. Gradio arayüzü kullanılamaz, dönüşümün Keras 3 modelleriyle çalıştığı ayrıca test edilmeli. Ciddi ek iş.
   - C) Gradio uygulamasını barındıran başka bir servis.
   - D) Canlı demo olmadan README'de ekran görüntüsü ya da GIF.
2. **Model dosyaları ve Git geçmişi:** `improvements` branch'inin geçmişinde birkaç model dosyası birikti:
   - `03b15df`: 22MB'lık eski MobileNetV2
   - `7902072`: 10MB'lık MobileNetV2
   - bu commit: 22MB'lık MobileNetV2
   - `main`'de zaten olan 80MB'lık orijinal `best_model.keras`

   HF Hub veya LFS'e geçerken bu dosyaların geçmişten temizlenip temizlenmeyeceğine karar verilmeli (ör. merge yerine squash). Güncel model dosyaları optimizer'sız (1.8MB ve 9.7MB), yani bundan sonraki commit'ler küçük olacak.
