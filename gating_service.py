"""
MelanomaLens — inference service for the GATED MobileNetV2 model (ISIC 2016 Part3B).

Self-contained: depends only on torch / torchvision / opencv / numpy / pytorch_grad_cam.
The clinical ABCD/TDS engine is vendored in below rather than imported from
`ai_service` — see the note on section 4.

Model
-----
MobileNetV2 (torchvision, plain ImageNet trunk — NO added attention) with a single
1x1-conv localization head (1281 params) that multiplicatively GATES the shared
feature map before pooling:

    F -> loc_head -> sigmoid -> A          (gate at feature resolution)
    F_gated = F * A
    logits  = classifier(pool(F_gated))

The gate is not decoration: the classifier physically cannot see regions the gate
marks as background, which is why this model's explanation map tracks the lesion
better than the same network's Grad-CAM (localization Dice 0.816 vs Grad-CAM IoU 0.4445).

Artifact expected at: models/GatedMobileNetV2_SE_ISIC3B_seed42_ft10k_ep1.pth (bare state_dict)

⚠ CLASS INDEX ORDER — this model is the OPPOSITE of the ViT model
-----------------------------------------------------------------
    this model : 0 = benign   , 1 = malignant
    ViT model  : 0 = melanoma , 1 = nevus      (see CLASS_LABELS in ai_service.py)

Copying `CLASS_LABELS` from `ai_service.py` without flipping the index silently
inverts every prediction (benign <-> malignant). Use the mapping defined HERE.

Preprocessing is the training-exact pipeline. Note it deliberately omits the
`cv2.MORPH_OPEN` step that `ai_service.remove_hair_dullrazor` performs, because
training never had it (see PREPROCESSING NOTE below).
"""
import base64
import os
import threading

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ==============================================================================
# 1. ARCHITECTURE  (must match training exactly: loc_experiment.GatedMobileNetV2)
# ==============================================================================

LOC_RES = 28            # localization map resolution used in training
TARGET_SIZE = 224
GATE_INIT_BIAS = 4.0


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block — byte-identical to the training-side R.SEBlock.

    Copied verbatim (architecture only) from run_pipeline_isic.SEBlock so the
    checkpoint's `attention.fc.*` keys load with strict=True. Channel reduction 16,
    no bias on either Linear, applied as a multiplicative re-weighting of the input.
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GatedMobileNetV2(nn.Module):
    """MobileNetV2 (+ optional SE attention) whose feature map is multiplicatively
    gated by a mask-supervised localization map.

    Architecture identical to the training-side `loc_experiment.GatedAttentionMobileNetV2`
    (SE trunk) / `GatedMobileNetV2` (no attention):

        F_att  = attention(features(x))        # 1280x7x7, attention optional
        A7     = sigmoid(loc_head(F_att))      # 1x1 conv, 1281 params  -> the gate
        logits = classifier(pool(F_att * A7))  # gating
        A      = sigmoid(upsample(loc_head(F_att), 28))   # Dice-supervised map

    `attention_type='SE'` is the deployed configuration: gate + SE trunk, lambda=1,
    gate head lr 3e-2 (sweep 3e-2 / 1e-1 / 3e-1 = 0.8150 / 0.8115 / 0.8083 test AUC).
    Keep it in sync with the checkpoint in models/ — a mismatch fails the strict load
    rather than silently running a different model.
    """

    def __init__(self, num_classes=2, loc_res=LOC_RES, gate_bias=GATE_INIT_BIAS,
                 pretrained=False, attention_type="SE"):
        super().__init__()
        # pretrained=False by default: the checkpoint overwrites every weight, and
        # downloading ImageNet weights at construction time would add a ~14 MB network
        # dependency to a cold start for no benefit.
        base = models.mobilenet_v2(weights=None) if not pretrained else \
            models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = base.features
        self.attention_type = attention_type
        self.attention = (SEBlock(1280) if attention_type == "SE" else None)
        if attention_type not in (None, "SE"):
            raise ValueError(f"unsupported attention_type: {attention_type}")
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_classes))
        self.loc_head = nn.Conv2d(1280, 1, kernel_size=1)
        nn.init.constant_(self.loc_head.bias, gate_bias)
        self.loc_res = loc_res

    def trunk(self, x):
        """Shared feature map F_att: the trunk output the gate and CAM both read."""
        f = self.features(x)
        return self.attention(f) if self.attention is not None else f

    def gate_at_feature_res(self, f):
        """A7 in [0,1]^(B,1,7,7) — the multiplicative gate."""
        return torch.sigmoid(self.loc_head(f))

    def forward(self, x, return_attn=False):
        f = self.trunk(x)
        a7 = self.gate_at_feature_res(f)
        f_gated = f * a7                       # gating
        logits = self.classifier(torch.flatten(self.pool(f_gated), 1))
        if not return_attn:
            return logits
        a = F.interpolate(self.loc_head(f), size=(self.loc_res, self.loc_res),
                          mode='bilinear', align_corners=False)
        return logits, torch.sigmoid(a)

    def localization_map(self, x):
        """A at loc_res (28x28) in [0,1], no_grad."""
        with torch.no_grad():
            f = self.trunk(x)
            a = F.interpolate(self.loc_head(f), size=(self.loc_res, self.loc_res),
                              mode='bilinear', align_corners=False)
            return torch.sigmoid(a)


def reshape_transform_vit(tensor, height=14, width=14):
    """Kept only so this module is import-compatible; the gating model is a CNN."""
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.transpose(2, 3).transpose(1, 2)


# ==============================================================================
# 2. CLASS MAPPING  (correct for THIS model — see module docstring)
# ==============================================================================

CLASS_TO_IDX = {'benign': 0, 'malignant': 1}
POSITIVE_IDX = CLASS_TO_IDX['malignant']          # index of the malignant class

CLASS_LABELS = {
    POSITIVE_IDX: {                                # 1 = malignant
        'label': 'Melanoma (Kanker Ganas)',
        'english': 'Malignant / Melanoma',
        'risk_level': 'Tinggi',
        'color': '#EF4444',
        'recommendation': ('Terdeteksi indikasi lesi ganas (Melanoma). Sangat disarankan '
                           'untuk segera berkonsultasi dengan Dokter Spesialis Dermatologi/Kulit.'),
    },
    CLASS_TO_IDX['benign']: {                      # 0 = benign
        'label': 'Nevus / Tahi Lalat (Jinak)',
        'english': 'Benign / Nevus',
        'risk_level': 'Rendah',
        'color': '#10B981',
        'recommendation': ('Lesi tampak jinak (non-kanker). Tetap lakukan pemantauan berkala '
                           'pada bentuk, batas, dan warna lesi.'),
    },
}

# ==============================================================================
# 3. PREPROCESSING  (training-exact)
# ==============================================================================

def remove_hair(img_rgb: np.ndarray):
    """DullRazor hair removal.

    PREPROCESSING NOTE: identical to the training pipeline's `remove_hair`
    (MelanomaProblemSolving/run_pipeline_isic.py). It is deliberately NOT identical
    to `ai_service.remove_hair_dullrazor`, which adds an extra `cv2.MORPH_OPEN 3x3`
    before the dilate. Training never had that step, so using it here would shift the
    input distribution away from what the weights were fitted on.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    k_size = max(9, min(17, min(gray.shape) // 30))
    k_size = k_size if k_size % 2 == 1 else k_size + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    blurred = cv2.GaussianBlur(blackhat, (3, 3), 0)
    _, mask = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)
    mask_d = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                        iterations=1)
    cleaned = cv2.inpaint(img_rgb, mask_d, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return cleaned, mask_d


def shades_of_grey(img: np.ndarray, power: float = 6.0) -> np.ndarray:
    img_f = img.astype(np.float32)
    illuminant = np.zeros(3)
    for i in range(3):
        val = np.mean(np.power(img_f[:, :, i], power))
        illuminant[i] = np.power(max(val, 1e-6), 1.0 / power)
    illuminant = np.maximum(illuminant, 1e-3)
    illuminant = illuminant / np.linalg.norm(illuminant) * np.sqrt(3)
    return np.clip(img_f / illuminant.reshape(1, 1, 3), 0, 255).astype(np.uint8)


def resize_and_pad(img_rgb: np.ndarray, target: int = TARGET_SIZE) -> np.ndarray:
    """Aspect-preserving resize (INTER_AREA) on the long side + centred zero padding."""
    h, w = img_rgb.shape[:2]
    scale = target / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    d_w, d_h = target - new_w, target - new_h
    top, bottom = d_h // 2, d_h - d_h // 2
    left, right = d_w // 2, d_w - d_w // 2
    return cv2.copyMakeBorder(resized, top, bottom, left, right,
                              borderType=cv2.BORDER_CONSTANT, value=0)


def to_tensor(prep_img: np.ndarray, dev) -> torch.Tensor:
    norm = (prep_img.astype(np.float32) / 255.0
            - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return torch.tensor(np.transpose(norm, (2, 0, 1)),
                        dtype=torch.float32).unsqueeze(0).to(dev)


def denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img_tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


# ==============================================================================
# 4. CONFIGURATION & STATE
# ==============================================================================

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
# Deployed model: gated MobileNetV2 with the SE trunk, lambda=1, gate head lr 3e-2,
# seed 42 (runs/loc_experiment/20260912_221206_lrsweep_lr003/arms/gate_se_lam1_seed42).
# Chosen actions on the 2026-09 study: best explanation + best malignant sensitivity of
# any arm measured on the clean split. Seed 42 (not the best-AUC seed 2026) so the model
# is not test-set selected and matches the previously deployed seed convention.
# Rollback: the previous plain-trunk artifact was REMOVED from models/ (2026-09 study
# cleanup). To go back, re-export it from the research arm
# runs/loc_experiment/20260912_033756_gatefix/arms/gate_lam1_seed42/weights/model.pth,
# drop it in models/, point MODEL_FILENAME at it and set attention_type=None -- the
# plain-trunk checkpoint has no attention.* keys, so the strict load fails loudly
# rather than running a mismatched model.
MODEL_FILENAME = 'GatedMobileNetV2_SE_ISIC3B_seed42_ft10k_ep1.pth'
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)
MODEL_DOWNLOAD_URL = ''      # in-repo artifact; no download required

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
cam_engine = None
model_loading = False
model_ready = False
model_load_error = None

# ==============================================================================
# 4. CLINICAL ABCD SEMIOLOGY & TDS ENGINE
#
# Vendored verbatim from ai_service.py so this module is genuinely self-contained.
# It is deliberately NOT imported from ai_service: when main.py switches to
# `import gating_service as ai_service`, a `from ai_service import ...` inside this
# module resolves to THIS module and fails, which silently dropped the whole ABCD
# block from the API response. The integration audit (Problem 3) lists a missing
# ABCD engine as a real defect, so it must never be silently optional.
# ==============================================================================

def compute_otsu_lesion_mask(img_rgb: np.ndarray) -> np.ndarray:
    """Segment the lesion with Otsu adaptive thresholding and morphological closing."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    return (cleaned > 127).astype(np.uint8)


def extract_asymmetry(mask_binary: np.ndarray):
    mask_u8 = (mask_binary > 0).astype(np.uint8)
    total_area = np.sum(mask_u8)
    if total_area == 0:
        return 0.0, 0, {}
    moments = cv2.moments(mask_u8)
    if moments['m00'] == 0:
        return 0.0, 0, {}
    cx = moments['m10'] / moments['m00']
    cy = moments['m01'] / moments['m00']
    mu20, mu02, mu11 = moments['mu20'], moments['mu02'], moments['mu11']
    theta_deg = float(np.degrees(0.5 * np.arctan2(2 * mu11, mu20 - mu02)))
    h, w = mask_u8.shape
    M_rot = cv2.getRotationMatrix2D((cx, cy), theta_deg, 1.0)
    aligned = cv2.warpAffine(mask_u8, M_rot, (w, h), flags=cv2.INTER_NEAREST)
    y_idx, x_idx = np.where(aligned > 0)
    if len(y_idx) == 0 or len(x_idx) == 0:
        return 0.0, 0, {}
    cropped = aligned[y_idx.min():y_idx.max() + 1, x_idx.min():x_idx.max() + 1]
    cropped_area = np.sum(cropped)
    mismatch_x = np.sum(np.abs(cropped.astype(float) - np.fliplr(cropped).astype(float))) / (2.0 * cropped_area)
    mismatch_y = np.sum(np.abs(cropped.astype(float) - np.flipud(cropped).astype(float))) / (2.0 * cropped_area)
    a_score = (1 if mismatch_x > 0.18 else 0) + (1 if mismatch_y > 0.18 else 0)
    return float(a_score * 1.3), a_score, {"mismatch_x": float(mismatch_x), "mismatch_y": float(mismatch_y)}


def extract_border_irregularity(mask_binary: np.ndarray):
    mask_u8 = (mask_binary > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0, 0, {}
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if area == 0 or perimeter == 0:
        return 0.0, 0, {}
    compactness = (perimeter ** 2) / (4.0 * np.pi * area)
    moments = cv2.moments(cnt)
    cx = moments['m10'] / moments['m00'] if moments['m00'] != 0 else mask_u8.shape[1] / 2
    cy = moments['m01'] / moments['m00'] if moments['m00'] != 0 else mask_u8.shape[0] / 2
    pts = cnt.squeeze()
    if len(pts.shape) < 2:
        return 0.0, 0, {}
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    radii = np.sqrt(dx ** 2 + dy ** 2)
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
    if np.sum(mask_bool) == 0:
        return 0.5, 1, {"detected_colors": ["Light Brown"]}
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lesion_hsv = hsv[mask_bool]
    H, S, V = lesion_hsv[:, 0] * 2, lesion_hsv[:, 1] / 255.0, lesion_hsv[:, 2] / 255.0
    total = len(H)
    min_pct = 0.03
    detected = {}
    if np.sum((V > 0.82) & (S < 0.22)) / total >= min_pct:
        detected["Putih (White)"] = True
    if np.sum((V < 0.22)) / total >= min_pct:
        detected["Hitam (Black)"] = True
    if np.sum(((H < 18) | (H > 342)) & (S > 0.30) & (V >= 0.22)) / total >= min_pct:
        detected["Merah (Red)"] = True
    if np.sum((H >= 170) & (H <= 265) & (S > 0.12) & (V >= 0.22) & (V <= 0.75)) / total >= min_pct:
        detected["Abu-kebiruan (Blue-Gray)"] = True
    if np.sum((H >= 10) & (H <= 35) & (S > 0.45) & (V >= 0.20) & (V < 0.55)) / total >= min_pct:
        detected["Cokelat Gelap (Dark Brown)"] = True
    if np.sum((H >= 15) & (H <= 45) & (S >= 0.22) & (S <= 0.65) & (V >= 0.50)) / total >= min_pct:
        detected["Cokelat Terang (Light Brown)"] = True
    c_score = max(1, min(6, len(detected)))
    return float(c_score * 0.5), c_score, {"detected_colors": list(detected.keys())}


def extract_diameter_and_structure(mask_binary: np.ndarray, contact_plate_mm: float = 20.0):
    mask_u8 = (mask_binary > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.5, 1, {"diameter_mm": 2.0}
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
    if tds < 4.75:
        category, risk = "Benign", "Risiko Rendah (Tahi Lalat Jinak)"
    elif tds <= 5.45:
        category, risk = "Suspicious", "Risiko Sedang (Lesi Perlu Pemantauan)"
    else:
        category, risk = "Malignant", "Risiko Tinggi (Kecurigaan Kuat Melanoma)"
    return {
        "a_score": a_raw, "b_score": b_raw, "c_score": c_raw, "d_score": d_raw,
        "diameter_mm": details_d["diameter_mm"], "detected_colors": details_c["detected_colors"],
        "tds": round(tds, 2), "clinical_category": category, "clinical_risk_level": risk,
    }


# ==============================================================================
# 5. MODEL LOADING
# ==============================================================================

def load_ai_model():
    global model, cam_engine, model_loading, model_ready, model_load_error
    from pytorch_grad_cam import GradCAM

    model_loading = True
    model_load_error = None

    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"{MODEL_PATH} not found. Place {MODEL_FILENAME} in the models/ directory.")

        print(f"[GatingModel] Loading {MODEL_FILENAME} on {device}...")
        # attention_type must match the artifact: 'SE' for the deployed SE-trunk
        # checkpoint, None for the older plain-trunk one (which has no attention.* keys).
        attention_type = "SE" if "_SE_" in MODEL_FILENAME else None
        net = GatedMobileNetV2(num_classes=2, loc_res=LOC_RES,
                               attention_type=attention_type)
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        # artifact is a bare state_dict, so a plain strict load is the contract
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        net.load_state_dict(state_dict, strict=True)
        net.to(device)
        net.eval()

        # Grad-CAM target: the same layer the research pipeline used, features[-1]
        # (1280 x 7 x 7). No reshape transform — this is a CNN, not a ViT.
        cam = GradCAM(model=net, target_layers=[net.features[-1]], reshape_transform=None)

        model = net
        cam_engine = cam
        model_ready = True
        model_loading = False
        print("[GatingModel] SUKSES! GatedMobileNetV2 & GradCAM loaded.")
        return True
    except Exception as e:
        model_load_error = str(e)
        model_loading = False
        print(f"[GatingModel] ERROR loading model: {e}")
        return False


def start_background_load():
    threading.Thread(target=load_ai_model, daemon=True, name="GatingModelLoader").start()


# ==============================================================================
# 6. EXPLANATIONS:  Grad-CAM  +  the model's own gate map
# ==============================================================================

def _attention_overlay(disp: np.ndarray, heat: np.ndarray, alpha: float = 0.65,
                       sharpen: float = 1.4, colormap: int = cv2.COLORMAP_JET,
                       contours_at: float = None) -> np.ndarray:
    """Blend a [0,1] attention map over an RGB image in [0,1], correctly.

    The map is used AS THE ALPHA, not as a fixed-weight layer. That distinction is
    the whole point: `cv2.COLORMAP_JET` maps 0 to dark blue RGB(0,0,128), so the
    common recipe `0.55*img + 0.45*heatmap` paints dark blue over every region the
    model did NOT attend — measured 27% of the frame blue-dominant for Grad-CAM and
    78% for the gate map, and it hides the lesion inside the padding. Here:

        out = img*(1 - a) + color*a     with a = alpha * heat**sharpen

    so where heat == 0 the output IS the input pixel, bit for bit. The `sharpen`
    exponent pushes weak, diffuse activation further down so only genuinely
    attended areas take colour.
    """
    h = heat.astype(np.float32)
    lo, hi = float(h.min()), float(h.max())
    if hi - lo > 1e-8:
        h = (h - lo) / (hi - lo)
    h = np.clip(h, 0.0, 1.0)
    a = np.clip(alpha * (h ** sharpen), 0.0, 1.0)[..., None]      # (H,W,1)

    color = cv2.applyColorMap(np.uint8(255 * h), colormap)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    out = disp * (1.0 - a) + color * a
    out = np.clip(out, 0.0, 1.0)
    if contours_at is not None:
        vis = (out * 255).astype(np.uint8).copy()
        cnts, _ = cv2.findContours((h > contours_at).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
        out = vis.astype(np.float32) / 255.0
    return out


def _png_data_url(rgb_uint8: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))
    return ("data:image/png;base64," + base64.b64encode(buf).decode("utf-8")) if ok else ""


def gradcam_overlay(prep_img: np.ndarray, tensor_x: torch.Tensor, target_class: int) -> str:
    """Grad-CAM heatmap over the preprocessed image.

    Target layer and CAM computation match the research pipeline (features[-1],
    1280x7x7); only the RENDERING differs — see _attention_overlay for why.
    """
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    try:
        grayscale_cam = cam_engine(input_tensor=tensor_x,
                                   targets=[ClassifierOutputTarget(target_class)])[0]
        disp = denormalize_image(tensor_x[0]).astype(np.float32)
        overlay = _attention_overlay(disp, grayscale_cam, alpha=0.65, sharpen=1.4)
        return _png_data_url((overlay * 255).astype(np.uint8))
    except Exception as e:
        print(f"[Grad-CAM Error]: {e}")
        return ""


def gate_map_overlay(tensor_x: torch.Tensor) -> tuple:
    """The model's own supervised localization map A, upsampled to 224 and blended.

    This is the map the Dice loss trained against the expert mask, so on this model
    it localises the lesion better than Grad-CAM does (Dice 0.816 at feature res).
    It also needs no backward pass.
    """
    try:
        with torch.no_grad():
            a = model.localization_map(tensor_x)                      # (1,1,28,28)
            a224 = F.interpolate(a, size=(TARGET_SIZE, TARGET_SIZE),
                                 mode='bilinear', align_corners=False)[0, 0]
        a_np = a224.cpu().numpy()
        disp = denormalize_image(tensor_x[0]).astype(np.float32)
        vis = _attention_overlay(disp, a_np, alpha=0.70, sharpen=1.6, contours_at=0.5)
        return _png_data_url((vis * 255).astype(np.uint8)), float((a_np > 0.5).mean()), float(a_np.mean())
    except Exception as e:
        print(f"[Gate Map Error]: {e}")
        return "", 0.0, 0.0


# ==============================================================================
# 7. INFERENCE  (drop-in compatible with ai_service.predict_lesion)
# ==============================================================================

def predict_lesion(image_bytes: bytes):
    global model, cam_engine, model_ready, model_loading, model_load_error

    if not model_ready:
        if model_loading:
            return {'status': 'error',
                    'message': 'Model AI sedang dimuat di background (~9MB). Silakan coba lagi sebentar.'}
        if not load_ai_model():
            return {'status': 'error', 'message': f'Gagal memuat model AI: {model_load_error}'}

    try:
        # 1. decode
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return {'status': 'error', 'message': 'Gagal mendekode berkas gambar.'}
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2. training-exact preprocessing
        cleaned, _ = remove_hair(img_rgb)
        sog = shades_of_grey(cleaned)
        prep_img = resize_and_pad(sog, TARGET_SIZE)
        tensor_x = to_tensor(prep_img, device)

        # 3. clinical ABCD / TDS engine (vendored above — always available; a failure
        #    here is a real error and should surface, not be silently skipped)
        lesion_mask = compute_otsu_lesion_mask(prep_img)
        abcd_result = compute_abcd_score(prep_img, lesion_mask)

        # 4. inference — index 1 is MALIGNANT for this model
        with torch.no_grad():
            logits = model(tensor_x)
            probabilities = torch.softmax(logits, dim=1)[0]
            prob_benign = float(probabilities[CLASS_TO_IDX['benign']].item())
            prob_malignant = float(probabilities[POSITIVE_IDX].item())

        predicted_class_idx = POSITIVE_IDX if prob_malignant >= 0.5 else CLASS_TO_IDX['benign']
        confidence = prob_malignant if predicted_class_idx == POSITIVE_IDX else prob_benign

        class_info = CLASS_LABELS[predicted_class_idx]
        is_malignant = (predicted_class_idx == POSITIVE_IDX)

        # 5. both explanations
        heatmap_b64 = gradcam_overlay(prep_img, tensor_x, target_class=predicted_class_idx)
        gate_b64, gate_area, gate_mean = gate_map_overlay(tensor_x)

        # 6. clinical-AI concordance
        concordance = None
        if abcd_result is not None:
            clinical_cat = abcd_result["clinical_category"]
            ai_cat = "Malignant" if is_malignant else "Benign"
            if clinical_cat == ai_cat:
                concordance = f"100% CONCORDANT (Keduanya Menunjukkan {clinical_cat})"
            elif clinical_cat == "Suspicious":
                concordance = f"BORDERLINE CONCORDANCE (TDS Meragukan, AI: {ai_cat})"
            else:
                concordance = f"DISCORDANT (TDS: {clinical_cat}, AI: {ai_cat})"

        out = {
            'status': 'success',
            'model_name': MODEL_FILENAME,
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
            'heatmap_base64': heatmap_b64,
            # --- added by the gating artifact ---
            'gate_map_base64': gate_b64,
            'gate_area_fraction': round(gate_area, 4),
            'gate_mean_activation': round(gate_mean, 4),
            'explanation': {
                'gradcam': 'Grad-CAM on features[-1] (1280x7x7), the pipeline standard',
                'gate_map': ('the model\'s own mask-supervised localization map A; on this model '
                             'it localises the lesion better than Grad-CAM (Dice 0.816 vs IoU 0.4445)'),
            },
            'class_mapping_note': 'index 0=benign, 1=malignant (OPPOSITE of the ViT model)',
        }
        if abcd_result is not None:
            out['abcd'] = {**abcd_result, 'concordance': concordance}
        return out
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'message': f'Gagal memproses gambar: {str(e)}'}
