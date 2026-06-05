"""Tekli goruntu tahmini (Phase 5).

Tek bir fotograf verildiginde:
1. Fotografu yukler ve model icin hazirlar
2. Egitilmis modelden gecirerek sinif tahmin eder
3. Sonuc: {sinif, guven_skoru, tum_olasiliklar} dondurur
"""
import json
import torch
from pathlib import Path
from PIL import Image

from .config import CONFIG, ROOT, select_device
from .model import build_model
from .data import build_transforms


_model = None
_classes = None
_transform = None
_device = None


def _load_model():
    """Modeli bir kez yukler, sonraki cagirilarda tekrar yuklemez."""
    global _model, _classes, _transform, _device

    if _model is not None:
        return

    _device = select_device()
    cfg_model = CONFIG["model"]
    cfg_paths = CONFIG["paths"]

    classes_path = ROOT / cfg_paths["classes_out"]
    model_path = ROOT / cfg_paths["model_out"]

    _classes = json.loads(classes_path.read_text(encoding="utf-8"))

    _model = build_model(cfg_model["architecture"], len(_classes), pretrained=False)
    _model.load_state_dict(torch.load(model_path, map_location=_device, weights_only=True))
    _model = _model.to(_device)
    _model.eval()

    _, val_tf = build_transforms(cfg_model["image_size"])
    _transform = val_tf


@torch.no_grad()
def predict_image(image_path: str) -> dict:
    """Tek fotograftan tahmin yapar.

    Returns:
        {
            "predicted_class": "lumpy_skin",
            "confidence": 0.923,
            "probabilities": {"healthy": 0.02, "lumpy_skin": 0.923, ...}
        }
    """
    _load_model()

    img = Image.open(image_path).convert("RGB")
    tensor = _transform(img).unsqueeze(0).to(_device)

    output = _model(tensor)
    probs = torch.softmax(output, dim=1).squeeze()

    conf, pred_idx = probs.max(dim=0)
    predicted_class = _classes[pred_idx.item()]

    probabilities = {cls: round(probs[i].item(), 4) for i, cls in enumerate(_classes)}

    return {
        "predicted_class": predicted_class,
        "confidence": round(conf.item(), 4),
        "probabilities": probabilities,
    }
