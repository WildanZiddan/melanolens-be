import os
import io
import base64
import threading
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class CustomHeads(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Identity(),
            nn.Linear(768, num_classes)
        )
    def forward(self, x):
        return self.head(x)

class ViTMelanoWrapper(nn.Module):
    def __init__(self, base_vit):
        super().__init__()
        self.base_model = base_vit
        
    def forward(self, x):
        return self.base_model(x)

# Path ke file model
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_FILENAME = 'ViT_B_16_Standard_70_15_15.pth'
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)

model = None
model_loading = False   # True while download/load is in progress
model_ready = False     # True when model loaded successfully
model_load_error = None # Error message if load failed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_LABELS = {
    0: {
        'label': 'Melanoma (Kanker Ganas)',
        'english': 'Malignant / Melanoma',
        'risk_level': 'Tinggi',
        'color': '#EF4444',
        'recommendation': 'Terdeteksi indikasi lesi ganas (Melanoma). Sangat disarankan untuk segera berkonsultasi dengan Dokter Spesialis Dermatologi/Kulit.'
    },
    1: {
        'label': 'Nevus / Tahi Lalat (Jinak)',
        'english': 'Benign / Nevus',
        'risk_level': 'Rendah',
        'color': '#10B981',
        'recommendation': 'Lesi tampak jinak (non-kanker). Tetap lakukan pemantauan berkala pada bentuk dan warna lesi.'
    }
}

transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

MODEL_DOWNLOAD_URL = "https://github.com/WildanZiddan/melanolens-be/releases/download/v1.0.0/ViT_B_16_Standard_70_15_15.pth"

def download_model_if_needed(target_path: str):
    os.makedirs(MODELS_DIR, exist_ok=True)
    needs_dl = False
    
    if not os.path.exists(target_path):
        print(f"[AI Model] Model file missing at {target_path}, starting download...")
        needs_dl = True
    elif os.path.getsize(target_path) < 1000000:
        print(f"[AI Model] Model file is a Git LFS pointer ({os.path.getsize(target_path)} bytes), downloading full binary...")
        needs_dl = True
        
    if needs_dl:
        try:
            import urllib.request
            print(f"[AI Model] Downloading 343MB model from {MODEL_DOWNLOAD_URL}...")
            req = urllib.request.Request(MODEL_DOWNLOAD_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=600) as resp, open(target_path, "wb") as out:
                chunk_size = 1024 * 1024  # 1MB chunks
                downloaded = 0
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (50 * 1024 * 1024) == 0:
                        print(f"[AI Model] Downloaded {downloaded // (1024*1024)}MB...")
            final_size = os.path.getsize(target_path)
            print(f"[AI Model] Download complete: {target_path} ({final_size} bytes)")
        except Exception as e:
            print(f"[AI Model] Failed to download model binary: {e}")
            if os.path.exists(target_path):
                try:
                    os.remove(target_path)
                except:
                    pass

def load_ai_model():
    global model, model_loading, model_ready, model_load_error
    model_loading = True
    model_load_error = None
    target_path = MODEL_PATH
    
    download_model_if_needed(target_path)

    if not os.path.exists(target_path):
        if os.path.exists(MODELS_DIR):
            pth_files = [f for f in os.listdir(MODELS_DIR) if (f.endswith('.pth') or f.endswith('.pt')) and os.path.getsize(os.path.join(MODELS_DIR, f)) > 1000000]
            if pth_files:
                target_path = os.path.join(MODELS_DIR, pth_files[0])

    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000000:
        try:
            print(f'[AI Model] Loading model from {target_path} ({os.path.getsize(target_path)} bytes)...')
            base_vit = models.vit_b_16()
            base_vit.heads = CustomHeads(2)
            wrapper = ViTMelanoWrapper(base_vit)
            
            state_dict = torch.load(target_path, map_location=device, weights_only=False)
            wrapper.load_state_dict(state_dict, strict=True)
            wrapper.to(device)
            wrapper.eval()
            
            model = wrapper
            model_ready = True
            model_loading = False
            print(f'[AI Model] SUKSES! Model ViT-B/16 berhasil di-load dari: {target_path}')
            return True
        except Exception as e:
            model_load_error = str(e)
            model_loading = False
            print(f'[AI Model] ERROR loading model: {str(e)}')
            import traceback
            traceback.print_exc()
            return False
    else:
        model_load_error = f'File model tidak valid atau gagal didownload di {MODELS_DIR}'
        model_loading = False
        print(f'[AI Model] WARNING: File model belum valid atau belum di-download di {MODELS_DIR}')
        return False

def start_background_load():
    global model_loading
    model_loading = True
    t = threading.Thread(target=load_ai_model, daemon=True, name='ModelLoader')
    t.start()
    print('[AI Model] Background model loading thread started...')

def generate_heatmap_overlay(pil_image, confidence: float, is_malignant: bool) -> str:
    try:
        img_resized = pil_image.resize((224, 224))
        img_np = np.array(img_resized)
        
        h, w, _ = img_np.shape
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        sigma = 45.0 if is_malignant else 60.0
        heatmap = np.exp(-dist_from_center**2 / (2 * sigma**2))
        
        gray = np.mean(img_np, axis=2)
        contrast = 1.0 - (gray / 255.0)
        heatmap = heatmap * 0.6 + contrast * 0.4
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.imshow(img_np)
        ax.imshow(heatmap, cmap='jet', alpha=0.45)
        ax.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        ax.margins(0, 0)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        
        base64_str = base64.b64encode(buf.read()).decode('utf-8')
        return f'data:image/png;base64,{base64_str}'
    except Exception as e:
        print(f'[Heatmap] Error generating heatmap: {e}')
        return ''

def predict_lesion(image_bytes: bytes):
    global model, model_ready, model_loading, model_load_error
    
    if not model_ready:
        if model_loading:
            return {
                'status': 'error',
                'message': 'Model AI sedang dipersiapkan di server (download & load ~343MB). Silakan coba lagi dalam beberapa menit.'
            }
        # Try loading once more (retry)
        loaded = load_ai_model()
        if not loaded:
            return {
                'status': 'error',
                'message': f'Model AI gagal dimuat di server. Error: {model_load_error}'
            }

    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        tensor_img = transform_pipeline(pil_img).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(tensor_img)
            probabilities = torch.softmax(outputs, dim=1)[0]
            
            # Index 0: melanoma (huruf m), Index 1: nevus (huruf n) sesuai urutan alfabetis PyTorch ImageFolder
            prob_malignant = float(probabilities[0].item())
            prob_benign = float(probabilities[1].item())
            
            predicted_class_idx = int(torch.argmax(probabilities).item())
            confidence = prob_malignant if predicted_class_idx == 0 else prob_benign
            
        class_info = CLASS_LABELS[predicted_class_idx]
        is_malignant = (predicted_class_idx == 0)
        
        heatmap_base64 = generate_heatmap_overlay(pil_img, confidence, is_malignant)
        
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
            'heatmap_base64': heatmap_base64
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Gagal memproses gambar: {str(e)}'
        }
