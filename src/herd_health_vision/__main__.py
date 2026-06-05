"""python -m herd_health_vision giris noktasi."""
import sys


def main():
    args = sys.argv[1:]

    if len(args) >= 2 and args[0] == "predict":
        from .predict import predict_image
        result = predict_image(args[1])
        print(f"\nTahmin: {result['predicted_class']}")
        print(f"Guven:  {result['confidence']:.2%}\n")
        for cls, prob in result["probabilities"].items():
            bar = "#" * int(prob * 30)
            print(f"  {cls:<18} {prob:.2%}  {bar}")
        print()
        return

    print("Herd Health Vision v0.1.0")
    print("Kullanim:")
    print("  python3 -m herd_health_vision predict <foto_yolu>")


if __name__ == "__main__":
    main()
