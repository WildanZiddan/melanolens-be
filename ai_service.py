import os
import io
import base64
import threading
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ==============================================================================
# 1. NEURAL ARCHITECTURE: CustomViT Matching Training Weights
# ==============================================================================

class CustomViT(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.base_model = models.vit_b_16(weights=None)
        in_features = self.base_model.heads.head.in_features
        self.base_model.heads.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

def reshape_transform_vit(tensor, height=14, width=14):
    """Reshape 196 patch tokens (excluding CLS token) back to 2D spatial grid (14x14)."""
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.transpose(2, 3).transpose(1, 2)

# ==============================================================================
# 2. CONFIGURATION & STATE
# ==============================================================================

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_FILENAME = 'ViT_B_16_Standard_70_15_15.pth'
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)
MODEL_DOWNLOAD_URL = "https://github.com/WildanZiddan/melanolens-be/releases/download/v1.0.0/ViT_B_16_Standard_70_15_15.pth"

TARGET_SIZE = 224
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = None
cam_engine = None
model_loading = False
model_ready = False
model_load_error = None

# 🔑 CORRECT CLASS MAPPING:
# Index 0: benign (Nevus / Tahi Lalat Jinak)
# Index 1: malignant (Melanoma Kanker Ganas)
CLASS_LABELS = {
    0: {
        'label': 'Nevus / Tahi Lalat (Jinak)',
        'english': 'Benign / Nevus',
        'risk_level': 'Rendah',
        'color': '#10B981',
        'recommendation': 'Lesi tampak jinak (non-kanker). Tetap lakukan pemantauan berkala pada bentuk, batas, dan warna lesi.'
    },
    1: {
        'label': 'Melanoma (Kanker Ganas)',
        'english': 'Malignant / Melanoma',
        'risk_level': 'Tinggi',
        'color': '#EF4444',
        'recommendation': 'Terdeteksi indikasi lesi ganas (Melanoma). Sangat disarankan untuk segera berkonsultasi dengan Dokter Spesialis Dermatologi/Kulit.'
    }
}

# ==============================================================================
# 3. MEDICAL PREPROCESSING PIPELINE
# ==============================================================================

def remove_hair_dullrazor(img_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Removes dark hair strands using BlackHat morphological filter and Telea inpainting."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    min_dim = min(h, w)
    k_size = max(9, min(17, min_dim // 30))
    k_size = k_size if k_size % 2 == 1 else k_size + 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    blurred = cv2.GaussianBlur(blackhat, (3, 3), 0)
    _, mask = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)

    k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_clean, iterations=1)
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_d = cv2.dilate(mask, k_dilate, iterations=1)

    cleaned = cv2.inpaint(img_rgb, mask_d, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return cleaned, mask_d

def shades_of_grey(img_rgb: np.ndarray, power: float = 6.0) -> np.ndarray:
    """Color constancy algorithm to normalize illumination across dermatoscopes."""
    img_f = img_rgb.astype(np.float32)
    illuminant = np.zeros(3)
    for i in range(3):
        val = np.mean(np.power(img_f[:, :, i], power))
        illuminant[i] = np.power(max(val, 1e-6), 1.0 / power)
    illuminant = np.maximum(illuminant, 1e-3)
    illuminant = illuminant / np.linalg.norm(illuminant) * np.sqrt(3)
    corrected = img_f / illuminant.reshape(1, 1, 3)
    return np.clip(corrected, 0, 255).astype(np.uint8)

def resize_and_pad(img_rgb: np.ndarray, target: int = 224) -> np.ndarray:
    """Preserves lesion aspect ratio and pads borders with black (matching training)."""
    h, w = img_rgb.shape[:2]
    scale = target / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    d_w, d_h = target - new_w, target - new_h
    top, bottom = d_h // 2, d_h - d_h // 2
    left, right = d_w // 2, d_w - d_w // 2
    return cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

# ==============================================================================
# 4. CLINICAL ABCD SEMIOLOGY & TDS ENGINE
# ==============================================================================

def compute_otsu_lesion_mask(img_rgb: np.ndarray) -> np.ndarray:
    """Segments skin lesion using Otsu adaptive thresholding and morphological closing."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    return (cleaned > 127).astype(np.uint8)

def extract_asymmetry(mask_binary: np.ndarray):
    mask_u8 = (mask_binary > 0).astype(np.uint8)
    total_area = np.sum(mask_u8)
    if total_area == 0: return 0.0, 0, {}
    moments = cv2.moments(mask_u8)
    if moments['m00'] == 0: return 0.0, 0, {}
    cx = moments['m10'] / moments['m00']
    cy = moments['m01'] / moments['m00']
    mu20, mu02, mu11 = moments['mu20'], moments['mu02'], moments['mu11']
    theta_deg = float(np.degrees(0.5 * np.arctan2(2 * mu11, mu20 - mu02)))
    h, w = mask_u8.shape
    M_rot = cv2.getRotationMatrix2D((cx, cy), theta_deg, 1.0)
    aligned = cv2.warpAffine(mask_u8, M_rot, (w, h), flags=cv2.INTER_NEAREST)
    y_idx, x_idx = np.where(aligned > 0)
    if len(y_idx) == 0 or len(x_idx) == 0: return 0.0, 0, {}
    cropped = aligned[y_idx.min():y_idx.max()+1, x_idx.min():x_idx.max()+1]
    cropped_area = np.sum(cropped)
    mismatch_x = np.sum(np.abs(cropped.astype(float) - np.fliplr(cropped).astype(float))) / (2.0 * cropped_area)
    mismatch_y = np.sum(np.abs(cropped.astype(float) - np.flipud(cropped).astype(float))) / (2.0 * cropped_area)
    a_score = (1 if mismatch_x > 0.18 else 0) + (1 if mismatch_y > 0.18 else 0)
    return float(a_score * 1.3), a_score, {"mismatch_x": float(mismatch_x), "mismatch_y": float(mismatch_y)}

def extract_border_irregularity(mask_binary: np.ndarray):
    mask_u8 = (mask_binary > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: return 0.0, 0, {}
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if area == 0 or perimeter == 0: return 0.0, 0, {}
    compactness = (perimeter ** 2) / (4.0 * np.pi * area)
    moments = cv2.moments(cnt)
    cx = moments['m10'] / moments['m00'] if moments['m00'] != 0 else mask_u8.shape[1] / 2
    cy = moments['m01'] / moments['m00'] if moments['m00'] != 0 else mask_u8.shape[0] / 2
    pts = cnt.squeeze()
    if len(pts.shape) < 2: return 0.0, 0, {}
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    radii = np.sqrt(dx**2 + dy**2)
    angles = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
    sector_scores = []
    for s in range(8):
        in_sec = (angles >= s * 45.0) & (angles < (s + 1) * 45.0)
        if np.sum(in_sec) > 3:
            s_rad = radii[in_sec]
            std_r = np.std(s_rad) / (np.mean(s_rad) + 1e-6)
            sector_scores.append(1 if std_r > 0.15 or (compactness > 1.6 and std_r > 0.10) else 0)
        else:
            sector_scores.append(1 if compactness > 1.8 else 0)
    b_score = min(8, sum(sector_scores))
    return float(b_score * 0.1), b_score, {"compactness": float(compactness)}

def extract_color_variegation(img_rgb: np.ndarray, mask_binary: np.ndarray):
    mask_bool = mask_binary > 0
    if np.sum(mask_bool) == 0: return 0.5, 1, {"detected_colors": ["Light Brown"]}
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lesion_hsv = hsv[mask_bool]
    H, S, V = lesion_hsv[:, 0] * 2, lesion_hsv[:, 1] / 255.0, lesion_hsv[:, 2] / 255.0
    total = len(H)
    min_pct = 0.03
    detected = {}
    if np.sum((V > 0.82) & (S < 0.22)) / total >= min_pct: detected["Putih (White)"] = True
    if np.sum((V < 0.22)) / total >= min_pct: detected["Hitam (Black)"] = True
    if np.sum(((H < 18) | (H > 342)) & (S > 0.30) & (V >= 0.22)) / total >= min_pct: detected["Merah (Red)"] = True
    if np.sum((H >= 170) & (H <= 265) & (S > 0.12) & (V >= 0.22) & (V <= 0.75)) / total >= min_pct: detected["Abu-kebiruan (Blue-Gray)"] = True
    if np.sum((H >= 10) & (H <= 35) & (S > 0.45) & (V >= 0.20) & (V < 0.55)) / total >= min_pct: detected["Cokelat Gelap (Dark Brown)"] = True
    if np.sum((H >= 15) & (H <= 45) & (S >= 0.22) & (S <= 0.65) & (V >= 0.50)) / total >= min_pct: detected["Cokelat Terang (Light Brown)"] = True
    c_score = max(1, min(6, len(detected)))
    return float(c_score * 0.5), c_score, {"detected_colors": list(detected.keys())}

def extract_diameter_and_structure(mask_binary: np.ndarray, contact_plate_mm: float = 20.0):
    mask_u8 = (mask_binary > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0.5, 1, {"diameter_mm": 2.0}
    cnt = max(contours, key=cv2.contourArea)
    _, radius = cv2.minEnclosingCircle(cnt)
    diameter_mm = (radius * 2.0 / mask_u8.shape[1]) * contact_plate_mm
    d_score = 5 if diameter_mm >= 6.0 else (4 if diameter_mm >= 5.0 else (3 if diameter_mm >= 4.0 else (2 if diameter_mm >= 3.0 else 1)))
    return float(d_score * 0.5), d_score, {"diameter_mm": float(round(diameter_mm, 2))}

def compute_abcd_score(img_rgb: np.ndarray, mask_binary: np.ndarray):
    w_a, a_raw, _ = extract_asymmetry(mask_binary)
    w_b, b_raw, _ = extract_border_irregularity(mask_binary)
    w_c, c_raw, details_c = extract_color_variegation(img_rgb, mask_binary)
    w_d, d_raw, details_d = extract_diameter_and_structure(mask_binary)
    tds = w_a + w_b + w_c + w_d
    if tds < 4.75: category, risk = "Benign", "Risiko Rendah (Tahi Lalat Jinak)"
    elif tds <= 5.45: category, risk = "Suspicious", "Risiko Sedang (Lesi Perlu Pemantauan)"
    else: category, risk = "Malignant", "Risiko Tinggi (Kecurigaan Kuat Melanoma)"
    return {
        "a_score": a_raw, "b_score": b_raw, "c_score": c_raw, "d_score": d_raw,
        "diameter_mm": details_d["diameter_mm"], "detected_colors": details_c["detected_colors"],
        "tds": round(tds, 2), "clinical_category": category, "clinical_risk_level": risk
    }

# ==============================================================================
# 5. MODEL LOADING & DOWNLOAD
# ==============================================================================

def download_model_if_needed(target_path: str):
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(target_path) or os.path.getsize(target_path) < 1000000:
        import urllib.request
        print(f"[AI Model] Downloading ViT model from {MODEL_DOWNLOAD_URL}...")
        req = urllib.request.Request(MODEL_DOWNLOAD_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=600) as resp, open(target_path, "wb") as out:
            chunk_size = 1024 * 1024
            while True:
                chunk = resp.read(chunk_size)
                if not chunk: break
                out.write(chunk)
        print(f"[AI Model] Download complete: {target_path}")

def load_ai_model():
    global model, cam_engine, model_loading, model_ready, model_load_error
    model_loading = True
    model_load_error = None
    target_path = MODEL_PATH

    download_model_if_needed(target_path)

    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000000:
        try:
            print(f"[AI Model] Instantiating CustomViT on {device}...")
            net = CustomViT(num_classes=2)
            state_dict = torch.load(target_path, map_location=device, weights_only=False)
            net.load_state_dict(state_dict, strict=True)
            net.to(device)
            net.eval()

            # Initialize real Grad-CAM for ViT
            target_layers = [net.base_model.encoder.layers[-1].ln_1]
            cam = GradCAM(model=net, target_layers=target_layers, reshape_transform=reshape_transform_vit)

            model = net
            cam_engine = cam
            model_ready = True
            model_loading = False
            print("[AI Model] SUKSES! CustomViT & GradCAM loaded successfully.")
            return True
        except Exception as e:
            model_load_error = str(e)
            model_loading = False
            print(f"[AI Model] ERROR loading model: {e}")
            return False
    else:
        model_load_error = "Model binary not found or corrupt."
        model_loading = False
        return False

def start_background_load():
    threading.Thread(target=load_ai_model, daemon=True, name="ModelLoader").start()

# ==============================================================================
# 6. INFERENCE & TRUE GRAD-CAM OVERLAY GENERATION
# ==============================================================================

def generate_true_gradcam_overlay(prep_img: np.ndarray, tensor_x: torch.Tensor, target_class: int) -> str:
    """Generates real Grad-CAM attention heatmap overlay as a Base64-encoded PNG."""
    global cam_engine
    try:
        heatmap = cam_engine(input_tensor=tensor_x, targets=[ClassifierOutputTarget(target_class)])[0]
        cam_resized = cv2.resize(heatmap, (TARGET_SIZE, TARGET_SIZE))
        cam_color = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)
        overlay = np.uint8(0.55 * prep_img + 0.45 * cam_color)

        is_success, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        if is_success:
            b64_str = base64.b64encode(buffer).decode("utf-8")
            return f"data:image/png;base64,{b64_str}"
        return ""
    except Exception as e:
        print(f"[Grad-CAM Error]: {e}")
        return ""

def predict_lesion(image_bytes: bytes):
    global model, cam_engine, model_ready, model_loading, model_load_error

    if not model_ready:
        if model_loading:
            return {'status': 'error', 'message': 'Model AI sedang dimuat di background (~343MB). Silakan coba lagi sebentar.'}
        if not load_ai_model():
            return {'status': 'error', 'message': f'Gagal memuat model AI: {model_load_error}'}

    try:
        # 1. Decode Raw Image Bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return {'status': 'error', 'message': 'Gagal mendekode berkas gambar.'}
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2. Medical Preprocessing (DullRazor + Shades of Grey + Padded 224x224)
        cleaned, _ = remove_hair_dullrazor(img_rgb)
        sog = shades_of_grey(cleaned)
        prep_img = resize_and_pad(sog, TARGET_SIZE)

        # 3. Clinical ABCD Rule & Total Dermatoscopy Score (TDS)
        lesion_mask = compute_otsu_lesion_mask(prep_img)
        abcd_result = compute_abcd_score(prep_img, lesion_mask)

        # 4. Standardized ImageNet Tensor Transformation
        norm_img = (prep_img.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        tensor_x = torch.tensor(np.transpose(norm_img, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(device)

        # 5. Deep Learning ViT Inference
        model.eval()
        with torch.no_grad():
            logits = model(tensor_x)
            probabilities = torch.softmax(logits, dim=1)[0]

            # 🔑 Index 0 = Benign, Index 1 = Malignant
            prob_benign = float(probabilities[0].item())
            prob_malignant = float(probabilities[1].item())

            predicted_class_idx = 1 if prob_malignant >= 0.5 else 0
            confidence = prob_malignant if predicted_class_idx == 1 else prob_benign

        class_info = CLASS_LABELS[predicted_class_idx]
        is_malignant = (predicted_class_idx == 1)

        # 6. Real Grad-CAM Attention Map
        heatmap_base64 = generate_true_gradcam_overlay(prep_img, tensor_x, target_class=predicted_class_idx)

        # 7. Clinical-AI Concordance Check
        clinical_cat = abcd_result["clinical_category"]
        ai_cat = "Malignant" if is_malignant else "Benign"
        if clinical_cat == ai_cat:
            concordance = f"100% CONCORDANT (Keduanya Menunjukkan {clinical_cat})"
        elif clinical_cat == "Suspicious":
            concordance = f"BORDERLINE CONCORDANCE (TDS Meragukan, AI: {ai_cat})"
        else:
            concordance = f"DISCORDANT (TDS: {clinical_cat}, AI: {ai_cat})"

        return {
            'status': 'success',
            'predicted_class_idx': predicted_class_idx,
            'label': class_info['label'],
            'english_label': class_info['english'],
            'risk_level': class_info['risk_level'],
            'confidence': round(confidence * 100, 2),
            'confidence_decimal': round(confidence, 4),
            'prob_benign': round(prob_benign * 100, 2),
            'prob_malignant': round(prob_malignant * 100, 2),
            'color': class_info['color'],
            'recommendation': class_info['recommendation'],
            'heatmap_base64': heatmap_base64,
            'abcd': {
                **abcd_result,
                'concordance': concordance
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'message': f'Gagal memproses gambar: {str(e)}'}
