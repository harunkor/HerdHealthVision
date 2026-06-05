"""Egitim dongusu (Phase 3)."""
import json
import torch
import torch.nn as nn
from collections import Counter
from pathlib import Path

from .config import CONFIG, ROOT, select_device
from .model import build_model
from .data import build_dataloaders


def _class_weights(train_dir: Path, classes: list[str], device: str) -> torch.Tensor:
    """Sinif dengesizligini telafi eden agirliklar hesaplar."""
    counts = []
    for cls in classes:
        d = train_dir / cls
        counts.append(sum(1 for f in d.iterdir() if f.is_file()))
    total = sum(counts)
    weights = [total / (len(counts) * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float32).to(device)


def _train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def _validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def run_training() -> dict:
    device = select_device()
    cfg_model = CONFIG["model"]
    cfg_train = CONFIG["training"]
    cfg_paths = CONFIG["paths"]

    train_dir = ROOT / cfg_paths["train"]
    val_dir = ROOT / cfg_paths["val"]
    model_out = ROOT / cfg_paths["model_out"]
    classes_out = ROOT / cfg_paths["classes_out"]
    model_out.parent.mkdir(parents=True, exist_ok=True)

    # Veri
    train_dl, val_dl, classes = build_dataloaders(
        str(train_dir), str(val_dir),
        cfg_model["image_size"], cfg_train["batch_size"], cfg_train["num_workers"],
    )
    num_classes = len(classes)
    print(f"Siniflar ({num_classes}): {classes}")
    print(f"Train: {len(train_dl.dataset)}  Val: {len(val_dl.dataset)}  Cihaz: {device}")

    # Model
    model = build_model(cfg_model["architecture"], num_classes, cfg_model["pretrained"])
    model = model.to(device)

    # Sinif agirlikli loss
    weights = _class_weights(train_dir, classes, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_train["learning_rate"])

    # Egitim dongusu
    best_val_acc = 0.0
    patience_counter = 0
    patience = cfg_train["early_stopping_patience"]
    epochs = cfg_train["epochs"]

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _train_one_epoch(model, train_dl, criterion, optimizer, device)
        val_loss, val_acc = _validate(model, val_dl, criterion, device)

        print(f"Epoch {epoch:>2}/{epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), model_out)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping: {patience} epoch boyunca iyilesme yok.")
                break

    # Sinif haritasini kaydet
    with open(classes_out, "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    print(f"\nModel: {model_out}")
    print(f"Siniflar: {classes_out}")
    print(f"En iyi val_acc: {best_val_acc:.3f}")

    return {"best_val_acc": best_val_acc, "model_path": str(model_out)}
