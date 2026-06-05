"""Herd Health Vision v1 — Gradio web interface."""
import json
import torch
import timm
import gradio as gr
from PIL import Image
from torchvision import transforms

# --- Model setup ---
MODEL_PATH = "herd_health_model.pt"
CLASSES_PATH = "classes.json"
ARCHITECTURE = "efficientnet_b2"
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

LABELS = {
    "healthy": "Healthy",
    "lumpy_skin": "Lumpy Skin Disease",
    "foot_and_mouth": "Foot-and-Mouth Disease",
    "mastitis": "Mastitis",
}

classes = json.loads(open(CLASSES_PATH, encoding="utf-8").read())
model = timm.create_model(ARCHITECTURE, pretrained=False, num_classes=len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def predict(image):
    if image is None:
        return {}
    img = image.convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).squeeze()
    return {LABELS.get(cls, cls): float(probs[i]) for i, cls in enumerate(classes)}


# --- Custom CSS ---
CSS = """
.hero-section {
    text-align: center;
    padding: 32px 20px 8px 20px;
}
.hero-title {
    font-size: 2.4em;
    font-weight: 800;
    background: linear-gradient(135deg, #1b5e20, #2e7d32, #43a047);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
    letter-spacing: -0.5px;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #2e7d32, #43a047);
    color: white;
    font-size: 0.7em;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    margin-left: 8px;
    vertical-align: middle;
    letter-spacing: 0.5px;
}
.hero-subtitle {
    font-size: 1.15em;
    color: #555;
    max-width: 680px;
    margin: 12px auto 0 auto;
    line-height: 1.6;
}
.how-it-works {
    display: flex;
    justify-content: center;
    gap: 32px;
    margin: 24px auto 8px auto;
    max-width: 600px;
    flex-wrap: wrap;
}
.step {
    text-align: center;
    flex: 1;
    min-width: 140px;
}
.step-icon {
    font-size: 1.8em;
    margin-bottom: 4px;
}
.step-num {
    font-size: 0.75em;
    font-weight: 700;
    color: #2e7d32;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.step-text {
    font-size: 0.9em;
    color: #555;
    margin-top: 2px;
}
.divider {
    border: none;
    border-top: 1px solid #e0e0e0;
    margin: 20px auto;
    max-width: 700px;
}
.disease-grid {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 14px;
    margin: 0 auto 12px auto;
    max-width: 750px;
}
.disease-card {
    border-radius: 12px;
    padding: 16px 20px;
    min-width: 155px;
    flex: 1;
    text-align: center;
    transition: transform 0.15s;
}
.disease-card:hover { transform: translateY(-2px); }
.card-healthy { background: linear-gradient(135deg, #e8f5e9, #c8e6c9); }
.card-lumpy { background: linear-gradient(135deg, #fff3e0, #ffe0b2); }
.card-fmd { background: linear-gradient(135deg, #fce4ec, #f8bbd0); }
.card-mastitis { background: linear-gradient(135deg, #e3f2fd, #bbdefb); }
.card-name { font-weight: 700; font-size: 0.95em; margin-bottom: 4px; }
.card-desc { font-size: 0.8em; color: #555; }
.info-panel {
    text-align: left;
    background: #fafafa;
    border: 1px solid #e8e8e8;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 14px;
    max-width: 750px;
    margin-left: auto;
    margin-right: auto;
}
.info-panel summary {
    cursor: pointer;
    font-weight: 700;
    font-size: 1em;
    color: #333;
}
.info-panel table { width: 100%; font-size: 0.88em; margin-top: 12px; border-collapse: collapse; }
.info-panel td, .info-panel th { padding: 6px 10px; border-bottom: 1px solid #eee; }
.info-panel th { text-align: left; color: #666; font-weight: 600; }
.disclaimer {
    background: #fff8e1;
    border-left: 4px solid #ffa000;
    padding: 14px 18px;
    border-radius: 8px;
    font-size: 0.88em;
    text-align: left;
    max-width: 750px;
    margin: 0 auto 8px auto;
    color: #555;
}
.footer {
    text-align: center;
    color: #aaa;
    font-size: 0.8em;
    margin-top: 24px;
    padding-bottom: 16px;
}
"""

# --- Build UI with Blocks ---
with gr.Blocks(theme=gr.themes.Soft(), css=CSS) as demo:

    # Hero section
    gr.HTML("""
    <div class="hero-section">
        <div class="hero-title">
            Herd Health Vision <span class="hero-badge">v1</span>
        </div>
        <p class="hero-subtitle">
            Detect cattle diseases early from a single photograph.
            Powered by deep learning, our model analyzes images of cattle and identifies
            signs of <strong>Lumpy Skin Disease</strong>, <strong>Foot-and-Mouth Disease</strong>,
            and <strong>Mastitis</strong> with up to <strong>91% precision</strong> per class &mdash; trained on <strong>5,334 images</strong>.
        </p>
        <div class="how-it-works">
            <div class="step">
                <div class="step-icon">1.</div>
                <div class="step-num">Upload</div>
                <div class="step-text">Drop or select a cattle photo</div>
            </div>
            <div class="step">
                <div class="step-icon">2.</div>
                <div class="step-num">Analyze</div>
                <div class="step-text">AI examines the image in seconds</div>
            </div>
            <div class="step">
                <div class="step-icon">3.</div>
                <div class="step-num">Result</div>
                <div class="step-text">Get diagnosis + confidence score</div>
            </div>
        </div>
    </div>
    """)

    # Image upload + prediction
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Cattle Image", height=360)
            btn = gr.Button("Analyze Image", variant="primary", size="lg")
        with gr.Column(scale=1):
            label_output = gr.Label(num_top_classes=4, label="Diagnosis Result")
            gr.HTML("""
            <div style="text-align:center; margin-top:10px;">
                <button onclick="
                    navigator.clipboard.writeText(window.location.href);
                    this.textContent='Link Copied!';
                    setTimeout(()=>this.textContent='Share This App', 2000);
                " style="
                    background: linear-gradient(135deg,#2e7d32,#43a047);
                    color:white; border:none; padding:10px 28px;
                    border-radius:8px; font-size:0.95em; font-weight:600;
                    cursor:pointer; transition:opacity 0.2s;
                ">Share This App</button>
            </div>
            """)

    btn.click(fn=predict, inputs=image_input, outputs=label_output)
    image_input.change(fn=predict, inputs=image_input, outputs=label_output)

    gr.HTML('<hr class="divider">')

    # Detectable conditions
    gr.HTML("""
    <div style="text-align:center; margin-bottom:12px;">
        <strong style="font-size:1.1em; color:#333;">Detectable Conditions</strong>
    </div>
    <div class="disease-grid">
        <div class="disease-card card-healthy">
            <div class="card-name" style="color:#2e7d32;">Healthy</div>
            <div class="card-desc">No visible disease indicators detected in the image</div>
        </div>
        <div class="disease-card card-lumpy">
            <div class="card-name" style="color:#e65100;">Lumpy Skin Disease</div>
            <div class="card-desc">Nodular dermatitis caused by LSDV, spread by arthropod vectors</div>
        </div>
        <div class="disease-card card-fmd">
            <div class="card-name" style="color:#c62828;">Foot-and-Mouth Disease</div>
            <div class="card-desc">Highly contagious vesicular disease caused by FMDV</div>
        </div>
        <div class="disease-card card-mastitis">
            <div class="card-name" style="color:#1565c0;">Mastitis</div>
            <div class="card-desc">Bacterial inflammation of the udder tissue</div>
        </div>
    </div>
    """)

    # Model details
    gr.HTML("""
    <details class="info-panel">
        <summary>Model Architecture & Training</summary>
        <table>
            <tr><td><strong>Architecture</strong></td><td>EfficientNet-B2 (timm library)</td></tr>
            <tr><td><strong>Parameters</strong></td><td>~9.1 million</td></tr>
            <tr><td><strong>Pre-training</strong></td><td>ImageNet-1K</td></tr>
            <tr><td><strong>Fine-tuning Dataset</strong></td><td>5,334 cattle images across 4 classes</td></tr>
            <tr><td><strong>Validation Accuracy</strong></td><td>89.4%</td></tr>
            <tr><td><strong>Input Resolution</strong></td><td>224 x 224 px</td></tr>
            <tr><td><strong>Optimizer</strong></td><td>Adam (lr = 1e-4) with class-weighted CrossEntropyLoss</td></tr>
            <tr><td><strong>Training Hardware</strong></td><td>Apple M4 GPU (MPS backend)</td></tr>
            <tr><td><strong>Data Sources</strong></td><td>Kaggle, Zenodo, Mendeley Data (CC BY 4.0)</td></tr>
        </table>
    </details>
    """)

    # Per-class performance
    gr.HTML("""
    <details class="info-panel">
        <summary>Per-Class Performance Metrics</summary>
        <table>
            <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>Val Samples</th></tr>
            <tr><td>Foot-and-Mouth Disease</td><td>0.912</td><td>0.916</td><td><strong>0.914</strong></td><td>239</td></tr>
            <tr><td>Healthy</td><td>0.886</td><td>0.865</td><td><strong>0.875</strong></td><td>414</td></tr>
            <tr><td>Lumpy Skin Disease</td><td>0.896</td><td>0.913</td><td><strong>0.905</strong></td><td>358</td></tr>
            <tr><td>Mastitis</td><td>0.833</td><td>0.882</td><td><strong>0.857</strong></td><td>34</td></tr>
        </table>
    </details>
    """)

    # Disclaimer
    gr.HTML("""
    <div class="disclaimer">
        <strong>Disclaimer:</strong> This tool is designed for <em>early screening purposes only</em>
        and does not constitute a veterinary diagnosis. Results may vary based on image quality,
        lighting conditions, camera angle, and cattle breed. Always consult a qualified veterinarian
        for clinical confirmation and treatment decisions.
    </div>
    """)

    # Footer
    gr.HTML("""
    <div class="footer">
        Herd Health Vision v1 &mdash; EfficientNet-B2 &bull; PyTorch &bull; Gradio<br>
        <a href="https://github.com/harunkor/HerdHealthVision" target="_blank" style="color:#888; text-decoration:none;">GitHub</a> &bull;
        Open-source datasets: Kaggle &bull; Zenodo &bull; Mendeley Data
    </div>
    """)

if __name__ == "__main__":
    demo.launch(share=True)
