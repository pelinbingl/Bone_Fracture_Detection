import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageFile
import gradio as gr
import cv2
import numpy as np

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

# ── CLAHE Preprocessing ──────────────────────────────────
def apply_clahe(image):
    # PIL → numpy
    img_array = np.array(image.convert("RGB"))
    
    # RGB → LAB renk uzayı (CLAHE sadece L kanalına uygulanır)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE uygula → kontrast normalize et
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # Kanalları birleştir → RGB'ye geri dön
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(img_enhanced)

# ── Test-Time Augmentation (TTA) ─────────────────────────
def predict_with_tta(tensor):
    preds = []
    
    # Orijinal
    with torch.no_grad():
        preds.append(torch.softmax(model(tensor), dim=1))
    
    # Yatay çevir
    flipped = torch.flip(tensor, dims=[3])
    with torch.no_grad():
        preds.append(torch.softmax(model(flipped), dim=1))
    
    # Hafif rotate (+10)
    rotated = transforms.functional.rotate(tensor, 10)
    with torch.no_grad():
        preds.append(torch.softmax(model(rotated), dim=1))
    
    # Hafif rotate (-10)
    rotated2 = transforms.functional.rotate(tensor, -10)
    with torch.no_grad():
        preds.append(torch.softmax(model(rotated2), dim=1))
    
    # Ortalama al
    return torch.stack(preds).mean(0)

# ── Tahmin fonksiyonu ────────────────────────────────────
def predict(image):
    if image is None:
        return {"🦴 Kırık (Fractured)": 0.0, "✅ Normal (Not Fractured)": 0.0}
    
    # CLAHE preprocessing
    img_enhanced = apply_clahe(image)
    
    # Transform
    tensor = transform(img_enhanced).unsqueeze(0)
    
    # TTA ile tahmin
    probs = predict_with_tta(tensor)[0]
    
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
        "**Test Accuracy: %99 | ROC-AUC: 0.9987**\n"
        "CLAHE preprocessing + Test-Time Augmentation ile domain shift azaltıldı."
    ),
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()