import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageFile
import gradio as gr

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Model yükle ──────────────────────────────────────────
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier[1].in_features, 2)
    )
    model.load_state_dict(torch.load("fracture_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ── Transform ────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

CLASSES = ["fractured", "not fractured"]

# ── Tahmin fonksiyonu ────────────────────────────────────
def predict(image):
    img = image.convert("RGB")
    tensor = transform(img).unsqueeze(0)  # batch boyutu ekle
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
    return {
        "🦴 Kırık (Fractured)":     float(probs[0]),
        "✅ Normal (Not Fractured)": float(probs[1]),
    }

# ── Gradio arayüzü ───────────────────────────────────────
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="X-Ray Görüntüsü Yükle"),
    outputs=gr.Label(num_top_classes=2, label="Tahmin"),
    title="🦴 Bone Fracture Detection",
    description=(
        "EfficientNet-B0 tabanlı transfer learning modeli ile "
        "X-ray görüntüsünden kemik kırığı tespiti.\n"
        "**Test Accuracy: %99 | ROC-AUC: 0.9987**"
    ),
    examples=[],  # HF'e yükledikten sonra örnek görüntü ekleyebilirsin
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()