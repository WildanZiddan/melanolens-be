from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Form
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
import datetime
import shutil
import os
import base64 

import models
import ai_service
from typing import Optional
from database import engine, get_db
from auth_utils import hash_password, verify_password, create_access_token

models.Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: jalankan model loading di background thread
    print("[Startup] Melanolens API starting up...")
    ai_service.start_background_load()
    yield
    # Shutdown
    print("[Shutdown] Melanolens API shutting down...")

app = FastAPI(title="Melanolens - API Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RegisterInput(BaseModel):
    nama: str
    email: EmailStr
    password: str
    tanggal_lahir: str
    jenis_kelamin: str
    pekerjaan: str

class LoginInput(BaseModel):
    email: EmailStr
    password: str

@app.post("/api/auth/register")
def register(data: RegisterInput, db: Session = Depends(get_db)):
    user_exists = db.query(models.MelMsUser).filter(models.MelMsUser.user_email == data.email).first()
    if user_exists:
        raise HTTPException(status_code=400, detail="Email sudah terdaftar, coba gunakan email lain!")
    
    dob = None
    if data.tanggal_lahir:
        try:
            dob = datetime.datetime.strptime(data.tanggal_lahir, "%Y-%m-%d")
        except ValueError:
            pass

    new_user = models.MelMsUser(
        user_nama=data.nama,
        user_email=data.email,
        user_password=hash_password(data.password),
        user_tanggalLahir=dob,
        user_jenisKelamin=data.jenis_kelamin,
        user_pekerjaan=data.pekerjaan,
        user_role="user"
    )
    db.add(new_user)
    db.commit()
    return {"status": "success", "message": "Pengguna berhasil didaftarkan!"}


class ResetPasswordInput(BaseModel):
    email: EmailStr
    new_password: str

@app.post("/api/auth/reset-password")
def reset_password(data: ResetPasswordInput, db: Session = Depends(get_db)):
    try:
        user = db.query(models.MelMsUser).filter(models.MelMsUser.user_email == data.email).first()
        if not user:
            raise HTTPException(status_code=404, detail="Email tidak ditemukan di sistem!")
        
        user.user_password = hash_password(data.new_password)
        db.commit()
        return {"status": "success", "message": "Password berhasil diperbarui! Silakan login dengan password baru."}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Gagal reset password: {str(e)}")

@app.post("/api/auth/login")
def login(data: LoginInput, db: Session = Depends(get_db)):
    try:
        user = db.query(models.MelMsUser).filter(models.MelMsUser.user_email == data.email).first()
        if not user:
            print(f"[LOGIN DEBUG] User not found for email: {data.email}")
            raise HTTPException(status_code=401, detail="Email atau password salah!")
            
        if not verify_password(data.password, user.user_password):
            print(f"[LOGIN DEBUG] Password mismatch for user: {data.email}")
            raise HTTPException(status_code=401, detail="Email atau password salah!")

        role = user.user_role if user.user_role else "user"
        nama = user.user_nama if user.user_nama else user.user_email

        token_payload = {
            "user_id": user.user_id,
            "email": user.user_email,
            "authority": [role]
        }
        
        token = create_access_token(data=token_payload)
        
        return {
            "status": "success",
            "token": token,
            "user": {
                "id": user.user_id,
                "name": nama,
                "email": user.user_email,
                "authority": [role],
                "tanggal_lahir": str(user.user_tanggalLahir) if user.user_tanggalLahir else "",
                "jenis_kelamin": user.user_jenisKelamin if user.user_jenisKelamin else "",
                "pekerjaan": user.user_pekerjaan if user.user_pekerjaan else ""
            }
        }
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"[LOGIN ERROR]: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Login Exception: {str(e)}")

    

class UpdateProfileInput(BaseModel):
    user_nama: str
    user_email: EmailStr
    user_tanggalLahir: str
    user_jenisKelamin: str
    user_pekerjaan: str

@app.put("/api/auth/update-profile")
def update_profile(data: UpdateProfileInput, db: Session = Depends(get_db)):
    user = db.query(models.MelMsUser).filter(models.MelMsUser.user_email == data.user_email).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="Pengguna tidak ditemukan!")
    
    user.user_nama = data.user_nama
    user.user_tanggalLahir = data.user_tanggalLahir
    user.user_jenisKelamin = data.user_jenisKelamin
    user.user_pekerjaan = data.user_pekerjaan
    
    db.commit()
    db.refresh(user)
    
    return {"status": "success", "message": "Data berhasil diupdate!"}

import base64



@app.get("/api/skrining/model-status")
def model_status():
    """Cek status model AI apakah sudah siap digunakan."""
    return {
        "model_ready": ai_service.model_ready,
        "model_loading": ai_service.model_loading,
        "model_load_error": ai_service.model_load_error,
        "device": str(ai_service.device),
    }

@app.post("/api/skrining/predict")
def predict_lesion_api(
    file: UploadFile = File(...),
    user_id: Optional[int] = Form(None),
    db: Session = Depends(get_db)
):
    try:
        image_bytes = file.file.read()
        prediction_result = ai_service.predict_lesion(image_bytes)
        
        if prediction_result.get("status") != "success":
            raise HTTPException(status_code=500, detail=prediction_result.get("message", "Gagal memproses gambar AI"))
            
        scan_id = None
        if user_id:
            try:
                base64_encoded = base64.b64encode(image_bytes).decode("utf-8")
                format_foto = f"data:{file.content_type};base64,{base64_encoded}"
                
                tds_str = f" [TDS: {prediction_result['abcd']['tds']}]" if "abcd" in prediction_result else ""
                new_scan = models.MelTrScan(
                    user_id=user_id,
                    scan_gambar=format_foto,
                    scan_persentase=prediction_result["confidence_decimal"],
                    scan_respon=f"{prediction_result['label']}{tds_str}",
                    scan_responGambar=prediction_result.get("heatmap_base64"),
                    scan_tanggal=datetime.datetime.utcnow()
                )
                db.add(new_scan)
                db.commit()
                db.refresh(new_scan)
                scan_id = new_scan.scan_id
            except Exception as db_err:
                print(f"[DB Save Error]: {db_err}")
                db.rollback()

        prediction_result["scan_id"] = scan_id
        return prediction_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediksi AI: {str(e)}")

@app.post("/api/skrining/save-scan")
def save_scan(
    user_id: int = Form(...), 
    persentase: float = Form(...), 
    respon: str = Form(...), 
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    try:
        gambar_biner = file.file.read()
        
        base64_encoded = base64.b64encode(gambar_biner).decode("utf-8")
        
        format_foto = f"data:{file.content_type};base64,{base64_encoded}"

        new_scan = models.MelTrScan(
            user_id=user_id,
            scan_gambar=format_foto,
            scan_persentase=persentase,                
            scan_respon=respon,                        
            scan_tanggal=datetime.datetime.utcnow()    
        )
        
        db.add(new_scan)
        db.commit()
        db.refresh(new_scan)
        
        return {
            "status": "success", 
            "message": "Lesi sudah dicek!",
            "scan_id": new_scan.scan_id,
            "scan_respon": new_scan.scan_respon,
            "scan_persentase": new_scan.scan_persentase
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Gagal enkripsi/simpan Base64: {str(e)}")

from typing import Optional


@app.get("/api/skrining/history")
def get_scan_history(user_id: str = None, db: Session = Depends(get_db)):
    try:
        if not user_id or user_id == "null" or user_id == "undefined" or user_id == "":
            return []
            
        try:
            val_user_id = int(user_id)
        except ValueError:
            return []

        scans = db.query(models.MelTrScan)\
                  .filter(models.MelTrScan.user_id == val_user_id)\
                  .order_by(models.MelTrScan.scan_tanggal.asc())\
                  .all()
                  
        return scans
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Gagal memuat rekam medis dari database: {str(e)}"
        )

@app.get("/api/admin/history")
def get_admin_all_history(db: Session = Depends(get_db)):
    try:
        results = db.query(
            models.MelTrScan.scan_id,
            models.MelTrScan.user_id,
            models.MelTrScan.scan_gambar,
            models.MelTrScan.scan_tanggal,
            models.MelTrScan.scan_persentase,
            models.MelTrScan.scan_respon,
            models.MelMsUser.user_nama
        ).join(
            models.MelMsUser, 
            models.MelTrScan.user_id == models.MelMsUser.user_id
        ).order_by(
            models.MelTrScan.scan_tanggal.asc()
        ).all()
        
        history_list = []
        for row in results:
            history_list.append({
                "scan_id": row.scan_id,
                "user_id": row.user_id,
                "user_nama": row.user_nama,
                "scan_gambar": row.scan_gambar,
                "scan_tanggal": row.scan_tanggal.isoformat() if row.scan_tanggal else "",
                "scan_persentase": row.scan_persentase,
                "scan_respon": row.scan_respon
            })
            
        return history_list
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Gagal menarik data seluruh riwayat: {str(e)}"
        )
        
import datetime
from sqlalchemy import func

import datetime
from sqlalchemy import func

@app.get("/api/admin/dashboard-stats")
def get_admin_dashboard_stats(db: Session = Depends(get_db)):
    try:
        now = datetime.datetime.utcnow()
        current_year = 2026
        
        start_of_week = (now - datetime.timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        start_of_month = datetime.datetime(now.year, now.month, 1)
        start_of_year = datetime.datetime(now.year, 1, 1)

        weekly_scan_series = [0] * 7
        weekly_malignant_series = [0] * 7
        weekly_benign_series = [0] * 7

        weekly_scans_raw = db.query(models.MelTrScan).filter(models.MelTrScan.scan_tanggal >= start_of_week).all()
        for scan in weekly_scans_raw:
            hari_index = scan.scan_tanggal.weekday()
            weekly_scan_series[hari_index] += 1
            if "melanoma" in scan.scan_respon.lower() or "ganas" in scan.scan_respon.lower():
                weekly_malignant_series[hari_index] += 1
            else:
                weekly_benign_series[hari_index] += 1

        monthly_scan_series = [0] * 4
        monthly_malignant_series = [0] * 4
        monthly_benign_series = [0] * 4

        monthly_scans_raw = db.query(models.MelTrScan).filter(models.MelTrScan.scan_tanggal >= start_of_month).all()
        for scan in monthly_scans_raw:
            tgl = scan.scan_tanggal.day
            w_idx = 0 if tgl <= 7 else 1 if tgl <= 14 else 2 if tgl <= 21 else 3
            monthly_scan_series[w_idx] += 1
            if "melanoma" in scan.scan_respon.lower() or "ganas" in scan.scan_respon.lower():
                monthly_malignant_series[w_idx] += 1
            else:
                monthly_benign_series[w_idx] += 1

        yearly_scan_series = [0] * 4
        yearly_malignant_series = [0] * 4
        yearly_benign_series = [0] * 4

        yearly_scans_raw = db.query(models.MelTrScan).filter(models.MelTrScan.scan_tanggal >= start_of_year).all()
        for scan in yearly_scans_raw:
            bln = scan.scan_tanggal.month
            q_idx = 0 if bln <= 3 else 1 if bln <= 6 else 2 if bln <= 9 else 3
            yearly_scan_series[q_idx] += 1
            if "melanoma" in scan.scan_respon.lower() or "ganas" in scan.scan_respon.lower():
                yearly_malignant_series[q_idx] += 1
            else:
                yearly_benign_series[q_idx] += 1

        total_scans = db.query(models.MelTrScan).count()
        total_malignant = db.query(models.MelTrScan).filter(
            func.lower(models.MelTrScan.scan_respon).like('%melanoma%') | 
            func.lower(models.MelTrScan.scan_respon).like('%ganas%')
        ).count()
        
        avg_confidence_tuple = db.query(func.avg(models.MelTrScan.scan_persentase)).first()
        avg_confidence = float(avg_confidence_tuple[0]) if avg_confidence_tuple[0] else 0.0

        all_scans_with_users = db.query(models.MelTrScan).join(
            models.MelMsUser, models.MelTrScan.user_id == models.MelMsUser.user_id
        ).all()

        age_young = 0
        age_product = 0
        age_elderly = 0

        for scan in all_scans_with_users:
            if scan.owner and scan.owner.user_tanggalLahir:
                user_year = scan.owner.user_tanggalLahir.year
                age = current_year - user_year
                if age < 25:
                    age_young += 1
                elif age <= 50:
                    age_product += 1
                else:
                    age_elderly += 1
            else:
                age_product += 1

        pct_young = round((age_young / total_scans) * 100) if total_scans > 0 else 30
        pct_product = round((age_product / total_scans) * 100) if total_scans > 0 else 50
        pct_elderly = round((age_elderly / total_scans) * 100) if total_scans > 0 else 20

        diagnosis_summary = [
            { "id": "1", "name": "Melanoma (Kanker Ganas)", "sales": total_malignant, "growShrink": 12.5 },
            { "id": "2", "name": "Nevus / Tahi Lalat (Jinak)", "sales": max(0, total_scans - total_malignant), "growShrink": -4.2 }
        ]

        male_count = db.query(models.MelTrScan).join(models.MelMsUser, models.MelTrScan.user_id == models.MelMsUser.user_id).filter(func.lower(models.MelMsUser.user_jenisKelamin).like('%laki%')).count()
        female_count = db.query(models.MelTrScan).join(models.MelMsUser, models.MelTrScan.user_id == models.MelMsUser.user_id).filter(func.lower(models.MelMsUser.user_jenisKelamin).like('%perempuan%')).count()
        male_pct = round((male_count / total_scans) * 100, 1) if total_scans > 0 else 0.0
        female_pct = round((female_count / total_scans) * 100, 1) if total_scans > 0 else 0.0

        recent_results = db.query(models.MelTrScan.scan_id, models.MelTrScan.scan_tanggal, models.MelTrScan.scan_respon, models.MelTrScan.scan_persentase, models.MelMsUser.user_nama).join(models.MelMsUser, models.MelTrScan.user_id == models.MelMsUser.user_id).order_by(models.MelTrScan.scan_tanggal.desc()).limit(5).all()
        recent_scans_list = []
        for row in recent_results:
            recent_scans_list.append({
                "scan_id": row.scan_id,
                "user_nama": row.user_nama,
                "scan_tanggal": row.scan_tanggal.strftime("%d %b %Y, %H:%M WIB") if row.scan_tanggal else "",
                "scan_respon": row.scan_respon,
                "scan_persentase": row.scan_persentase
            })

        return {
            "status": "success",
            "summary": {
                "weekly_scan": sum(weekly_scan_series),
                "weekly_malignant": sum(weekly_malignant_series),
                "weekly_benign": sum(weekly_benign_series),
                "monthly_scan": sum(monthly_scan_series),
                "monthly_malignant": sum(monthly_malignant_series),
                "monthly_benign": sum(monthly_benign_series),
                "yearly_scan": sum(yearly_scan_series),
                "yearly_malignant": sum(yearly_malignant_series),
                "yearly_benign": sum(yearly_benign_series),
                "avg_confidence": round(avg_confidence * 100, 1)
            },
            "charts": {
                "weekly_scan": weekly_scan_series,
                "weekly_malignant": weekly_malignant_series,
                "weekly_benign": weekly_benign_series,
                "monthly_scan": monthly_scan_series,
                "monthly_malignant": monthly_malignant_series,
                "monthly_benign": monthly_benign_series,
                "yearly_scan": yearly_scan_series,
                "yearly_malignant": yearly_malignant_series,
                "yearly_benign": yearly_benign_series
            },
            "gender_demographic": [
                { "id": "laki_laki", "name": "Laki-laki", "value": male_pct, "count": male_count },
                { "id": "perempuan", "name": "Perempuan", "value": female_pct, "count": female_count }
            ],
            "age_demographic": {
                "percentage": { "young": pct_young, "product": pct_product, "elderly": pct_elderly },
                "counts": { "young": age_young, "product": age_product, "elderly": age_elderly }
            },
            "diagnosis_summary": diagnosis_summary,
            "recent_scans": recent_scans_list
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal hitung statistik: {str(e)}")