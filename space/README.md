---
title: Herd Health Vision
emoji: 🐄
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: false
license: mit
---

# Herd Health Vision v1

AI-powered early screening tool for detecting common cattle diseases from photographs.

## Supported Conditions

| Class | Description |
|-------|-------------|
| **Healthy** | No disease indicators detected |
| **Lumpy Skin Disease** | Nodular dermatitis caused by Lumpy Skin Disease Virus (LSDV) |
| **Foot-and-Mouth Disease** | Highly contagious vesicular disease caused by FMDV |
| **Mastitis** | Inflammation or infection of the udder tissue |

## Model

- **Architecture:** EfficientNet-B2 (fine-tuned from ImageNet)
- **Training data:** 5,334 images across 4 classes
- **Validation accuracy:** 89.4%
- **Hardware:** Trained on Apple M4 (MPS)

## Disclaimer

This tool is intended for **pre-screening purposes only** and does not replace professional veterinary diagnosis. Always consult a qualified veterinarian for confirmed results.
