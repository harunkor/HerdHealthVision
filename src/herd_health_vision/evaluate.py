"""Degerlendirme modulu (Phase 4).

Egitilmis modeli val seti uzerinde test eder:
- Sinif bazinda precision, recall, F1
- Karisiklik matrisi (confusion matrix)
- En cok yanilan ornekler
- Sonuclari models/evaluation.json'a kaydeder
"""
import json
import torch
from collections import defaultdict
from pathlib import Path

from .config import CONFIG, ROOT, select_device
from .model import build_model
from .data import build_dataloaders


@torch.no_grad()
def run_evaluation() -> dict:
    device = select_device()
    cfg_model = CONFIG["model"]
    cfg_train = CONFIG["training"]
    cfg_paths = CONFIG["paths"]

    model_path = ROOT / cfg_paths["model_out"]
    classes_path = ROOT / cfg_paths["classes_out"]
    val_dir = ROOT / cfg_paths["val"]

    # Sinif listesini yukle
    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    num_classes = len(classes)

    # Model yukle
    model = build_model(cfg_model["architecture"], num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # Veri
    _, val_dl, _ = build_dataloaders(
        str(val_dir), str(val_dir),
        cfg_model["image_size"], cfg_train["batch_size"], cfg_train["num_workers"],
    )

    # --- Tahminleri topla ---
    all_preds = []
    all_labels = []
    all_confs = []

    for images, labels in val_dl:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        confs, preds = probs.max(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_confs.extend(confs.cpu().tolist())

    # --- Confusion matrix ---
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for true, pred in zip(all_labels, all_preds):
        matrix[true][pred] += 1

    # --- Sinif bazinda metrikler ---
    per_class = {}
    for i, cls in enumerate(classes):
        tp = matrix[i][i]
        fp = sum(matrix[j][i] for j in range(num_classes)) - tp
        fn = sum(matrix[i][j] for j in range(num_classes)) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = sum(matrix[i])
        per_class[cls] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": support,
        }

    # --- Genel metrikler ---
    total = len(all_labels)
    correct = sum(1 for t, p in zip(all_labels, all_preds) if t == p)
    accuracy = round(correct / total, 3)

    # --- Yanlis tahminler ---
    errors = []
    dataset = val_dl.dataset
    for idx in range(total):
        if all_preds[idx] != all_labels[idx]:
            path, _ = dataset.samples[idx]
            errors.append({
                "file": str(Path(path).relative_to(ROOT)),
                "true": classes[all_labels[idx]],
                "predicted": classes[all_preds[idx]],
                "confidence": round(all_confs[idx], 3),
            })
    errors.sort(key=lambda x: x["confidence"], reverse=True)

    # --- Sonuclari yazdir ---
    print(f"\n{'='*55}")
    print(f"  EVALUATION SONUCLARI  (val set: {total} goruntu)")
    print(f"{'='*55}")
    print(f"\n  Genel accuracy: {accuracy:.1%} ({correct}/{total})\n")

    print(f"  {'Sinif':<20}{'Prec':>8}{'Recall':>8}{'F1':>8}{'Adet':>8}")
    print(f"  {'-'*52}")
    for cls, m in per_class.items():
        print(f"  {cls:<20}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}{m['support']:>8}")

    print(f"\n  Karisiklik Matrisi (satir=gercek, sutun=tahmin):")
    header = "  " + " " * 18 + "".join(f"{c[:8]:>10}" for c in classes)
    print(header)
    for i, cls in enumerate(classes):
        row = "".join(f"{matrix[i][j]:>10}" for j in range(num_classes))
        print(f"  {cls:<18}{row}")

    if errors:
        print(f"\n  Yanlis tahminler ({len(errors)} adet, en yuksek guvenli ilk 10):")
        for e in errors[:10]:
            print(f"    {e['true']:>16} -> {e['predicted']:<16} guven={e['confidence']:.2f}  {e['file']}")

    print(f"\n{'='*55}\n")

    # --- JSON kaydet ---
    result = {
        "accuracy": accuracy,
        "total_samples": total,
        "correct": correct,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": classes,
            "matrix": matrix,
        },
        "errors_count": len(errors),
        "top_errors": errors[:20],
    }

    out_path = ROOT / "models" / "evaluation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Kaydedildi: {out_path}")

    return result
