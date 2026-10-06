---
title: Flower Species Classification
emoji: 🌷
colorFrom: pink
colorTo: green
sdk: gradio
sdk_version: 6.29.1
python_version: "3.11"
app_file: app.py
pinned: false
short_description: Classify 5 flower species with a CNN or MobileNetV2
---

# Flower Species Classification

Upload a flower photo and the model predicts one of five species: daisy, dandelion, rose, sunflower or tulip.
Both models were trained on the [`tf_flowers`](https://www.tensorflow.org/datasets/catalog/tf_flowers) dataset with TensorFlow / Keras.

| Model | Parameters | Test accuracy |
|---|---:|---:|
| MobileNetV2 (ImageNet weights, fine-tuned) — default | 2.26M | 93.45% |
| Custom CNN (4 conv blocks, BatchNorm, global average pooling) | 423K | 83.09% |

Accuracy is measured on a held-out test split of 550 images. The models were selected by validation loss.

The most common error for both models is confusing roses with tulips.

Source code and training details: [github.com/kubraaucar/CNN-Flower-Species-Classification](https://github.com/kubraaucar/CNN-Flower-Species-Classification)
