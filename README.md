# Herd Health Vision

Early screening tool for detecting common cattle diseases from photographs using deep learning. Built with transfer learning on EfficientNet-B2, trained on 5,334 images across 4 classes.

> **Disclaimer:** This is a pre-screening tool only. It does not replace professional veterinary diagnosis. Always consult a qualified veterinarian for clinical confirmation.

**Live Demo:** [huggingface.co/spaces/HARUNKOR/herd-health-vision](https://huggingface.co/spaces/HARUNKOR/herd-health-vision)

---

## Supported Conditions

| Class | Description |
|-------|-------------|
| **Healthy** | No visible disease indicators detected |
| **Lumpy Skin Disease** | Nodular dermatitis caused by Lumpy Skin Disease Virus (LSDV), spread by arthropod vectors |
| **Foot-and-Mouth Disease** | Highly contagious vesicular disease caused by FMDV, affects cloven-hoofed animals |
| **Mastitis** | Bacterial inflammation or infection of the udder tissue |

---

## Model Performance

**Overall validation accuracy: 89.4%** (934/1045 samples)

| Class | Precision | Recall | F1 Score | Validation Samples |
|-------|-----------|--------|----------|--------------------|
| Foot-and-Mouth Disease | 0.912 | 0.916 | **0.914** | 239 |
| Healthy | 0.886 | 0.865 | **0.875** | 414 |
| Lumpy Skin Disease | 0.896 | 0.913 | **0.905** | 358 |
| Mastitis | 0.833 | 0.882 | **0.857** | 34 |

### Confusion Matrix

```
                  Predicted
                  FMD    Healthy  Lumpy   Mastitis
Actual FMD        219    14       3       3
Actual Healthy    19     358      34      3
Actual Lumpy      2      29       327     0
Actual Mastitis   0      3        1       30
```

**Key observations:**
- Foot-and-Mouth Disease achieves the highest F1 score (0.914), with strong precision and recall balance
- Lumpy Skin Disease detection is robust (F1: 0.905), though some confusion with healthy cattle exists due to skin texture similarity
- Healthy classification shows solid precision (0.886); most errors involve borderline skin conditions misclassified as lumpy skin
- Mastitis has the smallest sample size (34 val images) but still achieves 88.2% recall, indicating the model learned meaningful udder inflammation patterns

---

## Model Architecture

| Property | Value |
|----------|-------|
| Architecture | EfficientNet-B2 ([timm](https://github.com/huggingface/pytorch-image-models)) |
| Parameters | ~9.1 million |
| Pre-training | ImageNet-1K |
| Fine-tuning strategy | Full model fine-tuning with pretrained weights |
| Input resolution | 224 x 224 px |
| Optimizer | Adam (lr = 1e-4) |
| Loss function | Class-weighted CrossEntropyLoss |
| Regularization | Early stopping (patience = 5) |
| Training hardware | Apple M4 GPU (MPS backend) |

### Training Data

| Class | Train | Validation | Total |
|-------|-------|------------|-------|
| Healthy | 1,775 | 414 | 2,189 |
| Lumpy Skin Disease | 1,347 | 358 | 1,705 |
| Foot-and-Mouth Disease | 1,031 | 239 | 1,270 |
| Mastitis | 136 | 34 | 170 |
| **Total** | **4,289** | **1,045** | **5,334** |

**Data sources:** Kaggle, Zenodo (CC BY 4.0 licensed datasets)

### Data Augmentation (Training)

- RandomResizedCrop (224x224)
- RandomHorizontalFlip
- ColorJitter (brightness, contrast, saturation)
- Normalization (ImageNet mean/std)

---

## Installation

### Requirements

- Python >= 3.10
- Apple Silicon (MPS), CUDA GPU, or CPU

### Setup

```bash
git clone https://github.com/harunkor/HerdHealthVision.git
cd HerdHealthVision
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### Single Image Prediction (CLI)

```bash
PYTHONPATH=src python -m herd_health_vision predict path/to/cattle_image.jpg
```

Example output:

```
Tahmin: lumpy_skin
Guven:  94.32%

  foot_and_mouth     1.22%
  healthy            3.15%  #
  lumpy_skin         94.32% ############################
  mastitis           1.31%
```

### Python API

```python
from herd_health_vision.predict import predict_image

result = predict_image("path/to/image.jpg")
print(result["predicted_class"])   # e.g. "lumpy_skin"
print(result["confidence"])        # e.g. 0.9432
print(result["probabilities"])     # dict of all class probabilities
```

### Web Interface (Gradio)

The app is deployed on Hugging Face Spaces. To run locally:

```bash
cd space
pip install -r requirements.txt
python app.py
```

---

## Project Structure

```
herd-health-vision/
├── config.yaml                  # All settings (classes, model, hyperparameters)
├── requirements.txt
├── pyproject.toml
├── LICENSE
├── data/
│   ├── train/<class>/           # Training images (not tracked in git)
│   └── val/<class>/             # Validation images (not tracked in git)
├── models/
│   ├── herd_health_model.pt     # Trained model weights (~30 MB, git-ignored)
│   ├── classes.json             # Class index mapping
│   └── evaluation.json          # Full evaluation metrics
├── space/                       # Hugging Face Spaces deployment
│   ├── app.py                   # Gradio web interface
│   ├── requirements.txt
│   ├── herd_health_model.pt     # Model copy for HF Spaces
│   └── classes.json
└── src/herd_health_vision/      # Core Python package
    ├── config.py                # Config loader + device selection (MPS/CUDA/CPU)
    ├── model.py                 # EfficientNet-B2 model builder
    ├── data.py                  # Data loading + augmentation pipeline
    ├── train.py                 # Training loop with class weighting + early stopping
    ├── evaluate.py              # Evaluation metrics + confusion matrix
    ├── predict.py               # Single-image inference
    └── __main__.py              # CLI entry point
```

---

## Training Your Own Model

1. **Prepare data** — Place images in `data/train/<class>/` and `data/val/<class>/` folders (80/20 split recommended)

2. **Configure** — Edit `config.yaml` to set your classes, model architecture, and hyperparameters

3. **Train:**
   ```bash
   PYTHONPATH=src python -c "from herd_health_vision.train import run_training; run_training()"
   ```

4. **Evaluate:**
   ```bash
   PYTHONPATH=src python -c "from herd_health_vision.evaluate import run_evaluation; run_evaluation()"
   ```

The trained model will be saved to `models/herd_health_model.pt` and evaluation results to `models/evaluation.json`.

---

## Improving Accuracy

Some strategies that helped improve accuracy from 84% to 89.4%:

- **Upgrade model architecture** — Moving from EfficientNet-B0 to B2 added more capacity
- **Expand dataset** — More diverse images from multiple sources improved generalization
- **Class-weighted loss** — Addresses class imbalance (e.g., mastitis has fewer samples)
- **Data augmentation** — Random crops, flips, and color jitter reduce overfitting
- **Early stopping** — Prevents training beyond the point of diminishing returns

---

## Notes

- `data/` and `*.pt` files are git-ignored due to size. Use [Git LFS](https://git-lfs.github.com/) if you need to track them.
- `models/classes.json` is small and tracked in git for inference support.
- The model auto-detects hardware: Apple Silicon MPS > NVIDIA CUDA > CPU.
- Mastitis class has limited training data (170 images). Adding more samples would likely improve its metrics.

---

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this software for any purpose.

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests. Some areas that could use help:

- Expanding the mastitis dataset for better class balance
- Adding new disease classes (e.g., bovine dermatophilosis, pink eye)
- Mobile-optimized inference pipeline
- Multi-language support for the web interface
