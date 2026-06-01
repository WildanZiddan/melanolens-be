from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr

import models
from database import engine, get_db
from auth_utils import hash_password, verify_password, create_access_token

# Perintah sakti otomatis bikin tabel ke cloud Supabase pas backend start
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Melanolens - API Backend")

# Mengizinkan Frontend Next.js (port 3000) buat nembak API tanpa diblokir CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schema validasi inputan dari frontend pake Pydantic
class RegisterInput(BaseModel):
    nama: str
    email: EmailStr
    password: str

class LoginInput(BaseModel):
    email: EmailStr
    password: str

@app.post("/api/auth/register")
def register(data: RegisterInput, db: Session = Depends(get_db)):
    user_exists = db.query(models.MelMsUser).filter(models.MelMsUser.user_email == data.email).first()
    if user_exists:
        raise HTTPException(status_code=400, detail="Email sudah terdaftar, coba gunakan email lain!")
    
    new_user = models.MelMsUser(
        user_nama=data.nama,
        user_email=data.email,
        user_password=hash_password(data.password),
        user_role="user" # Default role otomatis jadi user biasa / pasien
    )
    db.add(new_user)
    db.commit()
    return {"status": "success", "message": "Pengguna berhasil didaftarkan!"}

@app.post("/api/auth/login")
def login(data: LoginInput, db: Session = Depends(get_db)):
    user = db.query(models.MelMsUser).filter(models.MelMsUser.user_email == data.email).first()
    if not user or not verify_password(data.password, user.user_password):
        raise HTTPException(status_code=401, detail="Email atau password salah!")
    
    # Payload token disamakan strukturnya dengan array 'authority' di Next.js middleware lu
    token_payload = {
        "user_id": user.user_id,
        "email": user.user_email,
        "authority": [user.user_role]
    }
    
    token = create_access_token(data=token_payload)
    
    # 🔑 KONDISI BARU: Kirim semua data komplit dari tabel mel_msuser!
    return {
        "status": "success",
        "token": token,
        "user": {
            "name": user.user_nama,
            "email": user.user_email,
            "authority": [user.user_role],
            "tanggal_lahir": user.user_tanggalLahir if user.user_tanggalLahir else "",
            "jenis_kelamin": user.user_jenisKelamin if user.user_jenisKelamin else "",
            "pekerjaan": user.user_pekerjaan if user.user_pekerjaan else ""
        }
    }
    
# Schema untuk nangkep payload update profile dari Next.js
class UpdateProfileInput(BaseModel):
    user_nama: str
    user_email: EmailStr
    user_tanggalLahir: str
    user_jenisKelamin: str
    user_pekerjaan: str

@app.put("/api/auth/update-profile")
def update_profile(data: UpdateProfileInput, db: Session = Depends(get_db)):
    # 1. Cari data usernya di database berdasarkan email yang dikirim
    user = db.query(models.MelMsUser).filter(models.MelMsUser.user_email == data.user_email).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User tidak ditemukan!")
    
    # 2. Ganti nilai kolom di model SQLAlchemy dengan data baru dari frontend
    user.user_nama = data.user_nama
    user.user_tanggalLahir = data.user_tanggalLahir
    user.user_jenisKelamin = data.user_jenisKelamin
    user.user_pekerjaan = data.user_pekerjaan
    
    # 3. Eksekusi perintah sakti buat nge-push/save perubahan langsung ke cloud Supabase
    db.commit()
    db.refresh(user)
    
    return {"status": "success", "message": "Data berhasil diupdate ke Supabase!"}