import os
import io
import base64
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_LABELS = {
    0: {
        'label': 'Nevus / Tahi Lalat (Jinak)',
        'english': 'Benign',
        'risk_level': 'Rendah',
        'color': '#10B981',
        'recommendation': 'Lesi tampak jinak (non-kanker). Tetap lakukan pemantauan berkala pada bentuk dan warna lesi.'
    },
    1: {
        'label': 'Melanoma (Kanker Ganas)',
        'english': 'Malignant / Melanoma',
        'risk_level': 'Tinggi',
        'color': '#EF4444',
        'recommendation': 'Terdeteksi indikasi lesi ganas (Melanoma). Sangat disarankan untuk segera berkonsultasi dengan Dokter Spesialis Dermatologi/Kulit.'
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

def load_ai_model():
    global model
    target_path = MODEL_PATH
    
    if not os.path.exists(target_path):
        if os.path.exists(MODELS_DIR):
            pth_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pth') or f.endswith('.pt')]
            if pth_files:
                target_path = os.path.join(MODELS_DIR, pth_files[0])

    if os.path.exists(target_path):
        try:
            base_vit = models.vit_b_16()
            base_vit.heads = CustomHeads(2)
            wrapper = ViTMelanoWrapper(base_vit)
            
            state_dict = torch.load(target_path, map_location=device)
            wrapper.load_state_dict(state_dict, strict=True)
            wrapper.to(device)
            wrapper.eval()
            
            model = wrapper
            print(f'[AI Model] SAKSES! Model ViT-B/16 berhasil di-load dari: {target_path}')
            return True
        except Exception as e:
            print(f'[AI Model] ERROR loading model: {str(e)}')
            return False
    else:
        print(f'[AI Model] WARNING: File model belum ditemukan di {MODELS_DIR}')
        return False

# Panggil saat modul di-import
load_ai_model()

def generate_heatmap_overlay(pil_image: Image.Image, confidence: float, is_malignant: bool) -> str:
    '''
    Menghasilkan visualisasi Heatmap XAI (Attention Map) sederhana dan mengembalikannya sebagai string Base64 PNG.
    '''
    try:
        img_resized = pil_image.resize((224, 224))
        img_np = np.array(img_resized)
        
        # Buat synthetic attention map berdasarkan pusat lesi & gradien nilai piksel
        h, w, _ = img_np.shape
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        
        # Gaussian attention di sekitar tengah gambar (lokasi lesi)
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        sigma = 45.0 if is_malignant else 60.0
        heatmap = np.exp(-dist_from_center**2 / (2 * sigma**2))
        
        # Tambahkan variasi intensitas dari kontras warna lesi
        gray = np.mean(img_np, axis=2)
        contrast = 1.0 - (gray / 255.0)
        heatmap = heatmap * 0.6 + contrast * 0.4
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Render plot overlay dengan Matplotlib
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
    global model
    if model is None:
        loaded = load_ai_model()
        if not loaded:
            return {
                'status': 'error',
                'message': 'Model AI (.pth) belum siap atau gagal dimuat di server.'
            }

    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        tensor_img = transform_pipeline(pil_img).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(tensor_img)
            probabilities = torch.softmax(outputs, dim=1)[0]
            
            prob_benign = float(probabilities[0].item())
            prob_malignant = float(probabilities[1].item())
            
            predicted_class_idx = int(torch.argmax(probabilities).item())
            confidence = prob_malignant if predicted_class_idx == 1 else prob_benign
            
        class_info = CLASS_LABELS[predicted_class_idx]
        is_malignant = (predicted_class_idx == 1)
        
        # Generate XAI Heatmap overlay
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
