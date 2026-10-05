# İlerleme Notları

Branch: `improvements` · Son güncelleme: 2026-10-05

Tüm sonuçlar aynı test seti üzerinde: `tf_flowers` `train[85%:]`, 550 görüntü.
Ham metrikler ve confusion matrix'ler: `images/*_metrics.json`.

## Test sonuçları

| Model | Dosya | Parametre | Test Acc | Test Loss |
|---|---|---:|---:|---:|
| Baseline CNN (Flatten) | `baseline_model.keras` (git dışı) | 6.65M | %73.45 | 0.7170 |
| CNN + GAP | `best_model.keras` | 110K | %70.55 | 0.7525 |
| MobileNetV2 (TL + fine-tune) | `mobilenetv2_model.keras` | 2.26M | **%90.55** | **0.2321** |

### Sınıf bazlı F1 (precision / recall)

| Sınıf | Baseline | CNN + GAP | MobileNetV2 |
|---|---|---|---|
| Karahindiba | 0.78 (0.84 / 0.72) | 0.71 (0.70 / 0.72) | 0.95 (0.95 / 0.94) |
| Papatya | 0.74 (0.71 / 0.78) | 0.75 (0.72 / 0.78) | 0.91 (0.88 / 0.96) |
| Lale | 0.70 (0.70 / 0.70) | 0.70 (0.72 / 0.68) | 0.89 (0.86 / 0.93) |
| Ayçiçeği | 0.81 (0.74 / 0.90) | 0.81 (0.76 / 0.88) | 0.91 (0.94 / 0.89) |
| Gül | 0.64 (0.69 / 0.59) | 0.54 (0.60 / 0.49) | 0.86 (0.92 / 0.81) |

### Karışıklık sayıları

| | Baseline | CNN + GAP | MobileNetV2 |
|---|---:|---:|---:|
| Toplam hata | 146 | 162 | 52 |
| Gül → lale | 27 | 28 | 15 |
| Lale → gül | 19 | 26 | 7 |
| **Gül ↔ lale toplam (hataların payı)** | **46 (%32)** | **54 (%33)** | **22 (%42)** |
| Karahindiba → ayçiçeği | 16 | 19 | 2 |

## Yorumlar

- **GAP'li CNN underfit oluyor.** Eğitim accuracy'si yaklaşık %72, validation accuracy'si yaklaşık %69. Baseline'ın kapasitesi büyük ölçüde Flatten'dan sonraki 6.4M parametrelik Dense katmanındaydı. GAP bu katmanı ve konum bilgisini kaldırıyor. Üç conv bloğundan sonra her nöronun gördüğü alan yaklaşık 22×22 piksel, bu yüzden model yerel renk ve doku ortalamasıyla karar veriyor. Buna ek olarak ReduceLROnPlateau (patience 2), LR'yi 19–25. epoch'lar arasında 1e-3'ten 8e-6'ya düşürdü ve eğitim erken dondu. EarlyStopping 26. epoch'ta durdu, en iyi epoch 21 oldu.
- **Gül–lale en zor ayrım.** İki çiçeğin renk paleti ve kupa biçimli taç yaprakları benzer. MobileNetV2 bu hatayı 46'dan 22'ye indirdi, ama kalan hataların içindeki payı %32'den %42'ye çıktı. Kolay hatalar kayboldu, zor ayrım kaldı.
- **Karahindiba → ayçiçeği hatası 16'dan 2'ye indi.** CNN'ler büyük ölçüde "sarı" rengine bakıyordu. ImageNet'te öğrenilen şekil ve doku özellikleri bu hatayı neredeyse tamamen kaldırdı.
- **MobileNetV2:** 1. aşamada (gövde dondurulmuş) en iyi validation accuracy %88.6 idi. Fine-tuning sonrasında %91.3'e çıktı. Fine-tuning son epoch'ta da iyileşiyordu: validation loss 0.305'ten 0.274'e düştü ve EarlyStopping tetiklenmedi.

## Yapılan değişiklikler

- `cnn.py`: seed (42), `enable_op_determinism()`, shuffle seed'i, `Flatten` yerine `GlobalAveragePooling2D`, 40 epoch, EarlyStopping patience 3'ten 5'e. Grafikler ve metrikler `plt.show()` yerine `images/` klasörüne kaydediliyor.
- `transfer_learning.py` (yeni): MobileNetV2 (ImageNet, 224×224) kullanıyor. Ölçekleme ve veri artırma modelin içinde. Eğitim iki aşamalı: önce gövde dondurulmuş halde (lr 1e-3), sonra son 30 katman açılarak fine-tuning (lr 1e-5). Grafikler ve metrikler `images/` klasörüne kaydediliyor.
- `inference.py` / `app.py`: iki model destekleniyor. Her modelin girdi boyutu ve normalizasyonu ayrı tanımlı. Gradio'da model seçimi var ve varsayılan model MobileNetV2.
- `baseline_model.keras`: orijinal modelin yedeği. Boyutu 80MB, `.gitignore`'da.

## Adımlar

- [x] 1. Hazırlık: branch açıldı, baseline yedeklendi, seed eklendi
- [x] 2. GAP'li CNN eğitildi (sonuç: %70.55, bkz. açık karar 1)
- [~] 3. Grafikler ve confusion matrix kaydediliyor. Kalan: eksen etiketlerini İngilizce yapmak, gül–lale yorumunu README'ye yazmak
- [x] 4. MobileNetV2 transfer learning
- [~] 5. `inference.py` / `app.py` güncellendi. Kalan: gerçek bir görüntüyle iki modeli de test etmek
- [ ] 6. Model dosyası: LFS veya HF Hub, `requirements.txt`
- [ ] 7. Hugging Face Spaces deploy
- [ ] 8. README'yi İngilizce yeniden yazmak
- [ ] 9. GitHub About ve topic'ler, main'e merge

## Açık kararlar

1. **GAP'li CNN için ne yapılacak?**
   - A) Sonucu olduğu gibi bırakmak: "GAP parametreyi 60 kat azalttı ama sığ ağda underfit oldu."
   - B) Ara adım eklemek: 4. conv bloğu (256 filtre), BatchNorm ve ReduceLROnPlateau patience 3. Tablo Baseline → GAP → Derin GAP CNN → MobileNetV2 olur. CPU'da yaklaşık 30–40 dakika sürer.
2. **MobileNetV2 fine-tuning'i uzatılsın mı?** Son epoch'ta hâlâ iyileşiyordu. Epoch sayısını 10'dan 20'ye çıkarmak muhtemelen biraz daha kazanç sağlar.
3. **Model dosyaları:** `mobilenetv2_model.keras` 22MB, çünkü optimizer state'i içinde. Bu commit ile Git'e girdi. 6. adımda HF Hub seçilirse geçmişten temizlenmesi gerekebilir.
