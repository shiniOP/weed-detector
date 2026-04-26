from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
import torch
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F
from transformers import SwinModel
import io

app = FastAPI()

# ---------------- MODEL ----------------
class WeedDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = SwinModel.from_pretrained(
            'microsoft/swin-tiny-patch4-window7-224'
        )

        hidden_dim = self.backbone.config.hidden_size

        self.neck = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        self.cls_head  = nn.Linear(256, num_classes)
        self.bbox_head = nn.Linear(256, 4)

    def forward(self, imgs):
        imgs = F.interpolate(imgs, size=(224, 224), mode='bilinear', align_corners=False)

        out = self.backbone(pixel_values=imgs)

        # safer pooling
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feat = out.pooler_output
        else:
            feat = out.last_hidden_state.mean(dim=1)

        feat = self.neck(feat)

        cls_logits = self.cls_head(feat)
        bbox_pred  = self.bbox_head(feat).sigmoid()

        return cls_logits, bbox_pred


# ---------------- LOAD MODEL ----------------
device = torch.device('cpu')

model = WeedDetector()
model.load_state_dict(torch.load("best_weed_detector.pth", map_location=device))
model.eval()


# ---------------- PREPROCESS ----------------
IMG_SIZE = 416

def preprocess(image):
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
    return image


# ---------------- LABELS ----------------
CLASS_NAMES = {0: 'crop', 1: 'weed'}


# ---------------- JSON API ----------------
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    image = np.array(image)

    inp = preprocess(image)

    with torch.no_grad():
        cls_logits, bbox_pred = model(inp)

        pred = cls_logits.argmax(dim=1).item()
        confidence = torch.softmax(cls_logits, dim=1)[0][pred].item()

    return {
        "prediction": CLASS_NAMES[pred],
        "confidence": float(confidence)
    }


# ---------------- IMAGE API ----------------
@app.post("/predict_image/")
async def predict_image(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    image_np = np.array(image)

    h, w = image_np.shape[:2]

    inp = preprocess(image_np)

    with torch.no_grad():
        cls_logits, bbox_pred = model(inp)

        pred = cls_logits.argmax(dim=1).item()
        confidence = torch.softmax(cls_logits, dim=1)[0][pred].item()
        box = bbox_pred[0].cpu().numpy()

    # clamp box
    box = np.clip(box, 0, 1)

    x1 = int(box[0] * w)
    y1 = int(box[1] * h)
    x2 = int(box[2] * w)
    y2 = int(box[3] * h)

    # draw box
    color = (0, 255, 0) if pred == 0 else (255, 0, 0)

    cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)

    label = f"{CLASS_NAMES[pred]} {confidence*100:.1f}%"

    cv2.putText(
        image_np,
        label,
        (x1, max(y1 - 10, 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

    # encode image
    _, img_encoded = cv2.imencode('.jpg', image_np)

    return StreamingResponse(
        io.BytesIO(img_encoded.tobytes()),
        media_type="image/jpeg"
    )