"""Model factory: ImageNet'te eğitilmiş bir modeli fine-tuning için hazırlar."""
import timm


def build_model(architecture: str, num_classes: int, pretrained: bool = True):
    """
    Fine-tuning'e hazır model döndürür. 'Var olan modele fine-tuning' burada olur:
    pretrained ağırlıklar yüklenir, son katman num_classes'a göre değiştirilir.
    """
    return timm.create_model(
        architecture, pretrained=pretrained, num_classes=num_classes
    )
