# İlerleme Notları

Branch: `improvements` · Son güncelleme: 2026-10-06

Veri bölmesi (`tf_flowers`): eğitim `train[:70%]`, validation `train[70%:85%]` (551 görüntü), test `train[85%:]` (550 görüntü).
**Model seçimi validation loss'a göre yapılıyor, test seti sadece en sonda raporlanıyor.**
Ham metrikler ve confusion matrix'ler `images/*_metrics.json` dosyalarında. Dört modelin hepsi aynı script ile yeniden ölçüldü.

## Sonuçlar

| Model | Dosya | Parametre | Val Acc | Val Loss | Test Acc | Test Loss |
|---|---|---:|---:|---:|---:|---:|
| Baseline CNN (3 blok, Flatten) | `baseline_model.keras` (git dışı) | 6.65M | %72.78 | 0.766 | %73.45 | 0.717 |
| CNN + GAP (3 blok) | `cnn_gap_model.keras` | 110K | %68.06 | 0.810 | %70.55 | 0.753 |
| Derin CNN (4 blok, BatchNorm, GAP) | `best_model.keras` | 423K | %80.76 | 0.533 | %83.09 | 0.482 |
| MobileNetV2 (transfer learning) | `mobilenetv2_model.keras` | 2.26M | **%89.47** | **0.302** | **%89.64** | **0.278** |

### Sınıf bazlı F1 (precision / recall)

| Sınıf | Baseline | CNN + GAP | Derin CNN | MobileNetV2 |
|---|---|---|---|---|
| Karahindiba | 0.78 (0.84 / 0.72) | 0.71 (0.70 / 0.72) | 0.86 (0.82 / 0.91) | 0.94 (0.96 / 0.92) |
| Papatya | 0.74 (0.71 / 0.78) | 0.75 (0.72 / 0.78) | 0.83 (0.86 / 0.80) | 0.89 (0.83 / 0.96) |
| Lale | 0.70 (0.70 / 0.70) | 0.70 (0.72 / 0.68) | 0.82 (0.78 / 0.87) | 0.89 (0.88 / 0.90) |
| Ayçiçeği | 0.81 (0.74 / 0.90) | 0.81 (0.76 / 0.88) | 0.90 (0.91 / 0.88) | 0.90 (0.91 / 0.88) |
| Gül | 0.64 (0.69 / 0.59) | 0.54 (0.60 / 0.49) | 0.74 (0.80 / 0.68) | 0.87 (0.92 / 0.83) |

### Karışıklık sayıları (test)

| | Baseline | CNN + GAP | Derin CNN | MobileNetV2 |
|---|---:|---:|---:|---:|
| Toplam hata | 146 | 162 | 93 | 57 |
| Gül → lale | 27 | 28 | 21 | 11 |
| Lale → gül | 19 | 26 | 11 | 7 |
| **Gül ↔ lale toplam (hataların payı)** | **46 (%32)** | **54 (%33)** | **32 (%34)** | **18 (%32)** |
| Karahindiba → ayçiçeği | 16 | 19 | 4 | 3 |

## Yorumlar

- **Kendi CNN'imin gelişimi: %73.45 → %70.55 → %83.09.**
  - **GAP tek başına underfit oldu.** Eğitim accuracy'si yaklaşık %72, validation accuracy'si yaklaşık %69 kaldı. Baseline'ın kapasitesi büyük ölçüde Flatten'dan sonraki 6.4M parametrelik Dense katmanındaydı. Üç conv bloğundan sonra her nöronun gördüğü alan yaklaşık 22×22 piksel, bu yüzden model yerel renk ve doku ortalamasıyla karar veriyordu. Buna ek olarak ReduceLROnPlateau patience 2 olduğu için LR çok hızlı düştü.
  - **Derin CNN bu teşhisle tasarlandı.** 4. conv bloğu (256 filtre) görüş alanını genişletti. BatchNorm eğitimi hızlandırdı ve stabil hale getirdi. ReduceLROnPlateau patience'ı 3'e çıkarıldı. Model 32. epoch'ta durdu, en iyi epoch 27 oldu. Eğitim accuracy'si %83, validation accuracy'si %81, yani underfitting kalmadı. Sonuç baseline'dan 9.6 puan daha iyi ve parametre sayısı 16 kat daha az.
- **Gül–lale en zor ayrım.** Dört modelde de hataların yaklaşık üçte biri bu çiftten geliyor. Mutlak sayı 46'dan 18'e indi, ama hatalar içindeki payı değişmedi. İki çiçeğin renk paleti (kırmızı, pembe, sarı) ve kupa biçimli taç yaprakları benzer.
- **Karahindiba → ayçiçeği hatası 16'dan 4'e (derin CNN) ve 3'e (MobileNetV2) indi.** Sığ CNN'ler büyük ölçüde "sarı" rengine bakıyordu. Daha geniş görüş alanı ve ImageNet'te öğrenilen şekil özellikleri bu hatayı neredeyse tamamen kaldırdı.
- **MobileNetV2'de fine-tuning bu çalıştırmada seçilmedi.**
  - 1. aşamada (gövde dondurulmuş) en iyi validation loss 0.302 oldu (10. epoch).
  - Katmanlar açılınca eğitim loss'u 0.27'den 0.50'ye sıçradı ve validation loss 0.33–0.36'ya çıktı (bkz. `images/mobilenetv2_training_curves.png`).
  - EarlyStopping (patience 3) fine-tuning'i 4 epoch sonra durdurdu. 20 epoch üst sınırı hiç devreye girmedi.
  - Validation'a göre seçim yapıldığı için kaydedilen model 1. aşamanın modeli oldu.
  - Önceki çalıştırmada da fine-tuning'in ilk 3 epoch'u 1. aşamanın en iyi sonucundan kötüydü (0.325, 0.336, 0.327, karşılaştırma değeri 0.308). O seferki model, 4. epoch'ta EarlyStopping'in kendi sayacındaki ilk değeri tesadüfen geçtiği için devam edebilmişti. Yani katmanları açtıktan sonraki 3–4 epoch'luk bozulma tutarlı bir desen, patience 3 ise bunu atlatmak için sınırda kalıyor.

### Önceki MobileNetV2 çalıştırması (referans)

- Bu çalıştırma determinism olmadan ve CNN ile aynı anda koşmuştu. Fine-tuning seçilmişti.
- Sonuçları: validation accuracy %91.29, validation loss 0.274, test accuracy %90.55, test loss 0.232.
- Model dosyası diskte yeni modelle ezildi, ama `03b15df` commit'inde duruyor.

## Yapılan değişiklikler

- `cnn.py`:
  - Seed (42), `enable_op_determinism()` ve shuffle seed'i.
  - 4 conv bloğu (32, 64, 128 ve 256 filtre), her blok Conv → BatchNorm → ReLU → MaxPool sırasıyla.
  - `GlobalAveragePooling2D`, en fazla 40 epoch, EarlyStopping patience 5, ReduceLROnPlateau patience 3.
  - Validation ve test metrikleri ile grafikler `images/cnn_deep_*` dosyalarına kaydediliyor.
- `transfer_learning.py`:
  - MobileNetV2 (ImageNet, 224×224). Ölçekleme ve veri artırma modelin içinde.
  - 1. aşama: gövde dondurulmuş, lr 1e-3, 10 epoch.
  - 2. aşama: son 30 katman açılıyor, lr 1e-5, en fazla 20 epoch, EarlyStopping patience 3.
  - Seed, op determinism ve shuffle seed'i eklendi.
- `inference.py` / `app.py`: iki model destekleniyor ve her modelin girdi boyutu ile normalizasyonu ayrı tanımlı. Varsayılan model MobileNetV2, validation'a göre en iyi model de o.
- Model dosyaları:
  - `baseline_model.keras` (80MB): git dışında.
  - `cnn_gap_model.keras`: GAP modelinin yedeği.
  - `best_model.keras`: artık derin CNN'i tutuyor.
- Eğitim log'ları `logs/` klasöründe ve git dışında. `.claude/settings.local.json` git dışında ve Claude attribution'ı kapalı.

## Adımlar

- [x] 1. Hazırlık: branch açıldı, baseline yedeklendi, seed eklendi
- [x] 2. CNN iyileştirme: GAP (%70.55), ardından derin CNN (%83.09)
- [~] 3. Grafikler ve confusion matrix kaydediliyor, yeni grafiklerin etiketleri İngilizce. Kalan: gül–lale yorumunu README'ye yazmak
- [~] 4. MobileNetV2 tamamlandı. Fine-tuning kararı açık (bkz. açık karar 1)
- [~] 5. `inference.py` / `app.py` güncellendi. Kalan: gerçek bir görüntüyle iki modeli de test etmek
- [ ] 6. Model dosyası: LFS veya HF Hub, `requirements.txt`
- [ ] 7. Hugging Face Spaces deploy
- [ ] 8. README'yi İngilizce yeniden yazmak
- [ ] 9. GitHub About ve topic'ler, main'e merge

## Açık kararlar

1. **MobileNetV2 fine-tuning'i:**
   - A) 1. aşama modelini olduğu gibi kullanmak. Test sonucu %89.64.
   - B) Fine-tuning'deki EarlyStopping patience'ını 5–6'ya çıkarıp yeniden eğitmek. Katmanları açtıktan sonraki bozulmayı atlatma şansı verir. CPU'da yaklaşık 40 dakika sürer.
   - C) Önceki çalıştırmanın modelini geri yüklemek (validation loss 0.274, şu anki modelden daha iyi). Ama bu model mevcut kodla tekrar üretilemiyor.
2. **Model dosyaları ve Git geçmişi:** `03b15df` commit'inde 22MB'lık eski `mobilenetv2_model.keras` var. 6. adımda HF Hub seçilirse geçmişten temizlenmesi gerekebilir.
