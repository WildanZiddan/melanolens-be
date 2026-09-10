# 🔬 Melanolens - Panduan Pengujian Lokal (Dosen / Penguji)

Sistem Deteksi Lesi Kulit & Kanker Melanoma Berbasis Deep Learning **Vision Transformer (ViT-B/16)** dan Antarmuka Web Modern (Next.js & FastAPI).

---

## 📋 1. Ringkasan Arsitektur Sistem

- **Frontend**: Next.js 16 (React 19, Tailwind CSS, Turbopack, Framer Motion)
- **Backend API**: FastAPI (Python 3.10+ / PyTorch / Torchvision / SQLAlchemy)
- **Model AI**: Vision Transformer (**ViT-B/16**) Fine-Tuned untuk Klasifikasi Biner (*Benign / Nevus* vs *Malignant / Melanoma*) dengan fitur visualisasi Heatmap Attention/Grad-CAM.
- **Database**: PostgreSQL (Supabase Cloud Database dengan failover pooler port 6543/5432 & SQLite fallback).

---

## 🚀 2. Cara Menjalankan Backend (FastAPI + AI Model)

### Langkah-langkah:
1. Buka Terminal / CMD / PowerShell di folder `melanolens-be/`.
2. Buat & aktifkan virtual environment (jika belum ada):
   ```bash
   # Windows:
   python -m venv .venv
   .venv\Scripts\activate

   # Linux / macOS:
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Jalankan backend server:
   ```bash
   uvicorn main:app --host 127.0.0.1 --port 8000
   ```
5. Server akan berjalan di:
   - **Base URL**: `http://127.0.0.1:8000`
   - **Interactive API Documentation (Swagger UI)**: `http://127.0.0.1:8000/docs`
   - **Status Model AI**: `http://127.0.0.1:8000/api/skrining/model-status`

> **Catatan Model AI**:
> File model `ViT_B_16_Standard_70_15_15.pth` (343 MB) tersimpan di dalam folder `models/`. Jika menggunakan versi *lightweight*, sistem secara otomatis mengunduh model langsung dari GitHub Release pada saat aplikasi pertama kali dijalankan.

---

## 💻 3. Cara Menjalankan Frontend (Next.js)

### Langkah-langkah:
1. Buka Terminal / CMD baru di folder `melanolens/`.
2. Install dependensi Node.js:
   ```bash
   npm install
   ```
3. Jalankan server pengembangan (dev):
   ```bash
   npm run dev
   ```
4. Buka browser di:
   - **URL Web**: `http://localhost:3000`

---

## 🔑 4. Akun Login untuk Pengujian

Gunakan akun berikut untuk masuk ke dalam aplikasi:

| Role | Email | Password |
| :--- | :--- | :--- |
| **Admin** | `wildanziddan28@gmail.com` | `password123` |
| **Pasien / User** | `mohrafifabdilah@gmail.com` | `password123` |

*(Penguji juga dapat mendaftarkan akun baru melalui halaman Register/Sign-up).*

---

## 🧪 5. Alur Pengujian Fitur Utama

1. **Login**:
   - Masuk ke `http://localhost:3000/sign-in` menggunakan akun di atas.
2. **Skrining / Deteksi AI**:
   - Buka menu **Skrining / Scan** (`http://localhost:3000/dashboards/scan` atau `http://localhost:3000/home/scan`).
   - Unggah gambar lesi kulit / tahi lalat (format JPG/PNG).
   - Klik tombol **Mulai Analisis / Prediksi**.
   - Model ViT-B/16 akan menganalisis lesi, menampilkan:
     - Hasil Diagnosis (*Melanoma* atau *Nevus/Jinak*)
     - Skor Keyakinan (*Confidence Score* dalam %)
     - Peta Panas (*Attention Heatmap Overlay*)
     - Rekomendasi Medis Awal
3. **Riwayat Medis**:
   - Riwayat scan otomatis tersimpan ke database dan dapat dilihat di halaman **Riwayat / History**.
4. **Dashboard Statistik (Admin)**:
   - Buka halaman Dashboard untuk melihat ringkasan statistik scan mingguan, bulanan, tahunan, serta demografi usia dan gender pasien.

---

## 📡 6. Endpoint Utama API Backend

| Metode | Endpoint | Deskripsi |
| :--- | :--- | :--- |
| `GET` | `/api/skrining/model-status` | Cek kesiapan model AI di memori server |
| `POST` | `/api/auth/login` | Autentikasi dan penerbitan JWT token |
| `POST` | `/api/auth/register` | Pendaftaran pengguna baru |
| `POST` | `/api/skrining/predict` | Prediksi citra lesi kulit menggunakan ViT-B/16 |
| `GET` | `/api/skrining/history` | Mengambil riwayat scan pengguna |
| `GET` | `/api/admin/dashboard-stats` | Mengambil data analitik dan statistik untuk dashboard admin |
