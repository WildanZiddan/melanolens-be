from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Gunakan DIRECT_URL dari Supabase, tambahkan ?sslmode=require di ujungnya
DATABASE_URL = "postgresql://postgres.mfxgrtogcmmxpggtvalk:Melanolens_MuhammadRidha@aws-1-ap-southeast-1.pooler.supabase.com:5432/postgres?sslmode=require"

# 🔑 SOLUSI SAKTI: Tambahin konfigurasi pooling biar gak putus nyambung kayak hubungan lu wkwk
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,    # 🕵️‍♂️ Cek koneksi dulu sebelum query. Kalau Supabase mutusin sepihak, SQLAlchemy otomatis bikin koneksi baru!
    pool_recycle=1800,     # 🔄 Reset koneksi secara berkala tiap 30 menit biar gak basi
    pool_size=10,          # Batas pool koneksi yang ditampung
    max_overflow=20        # Toleransi tambahan kalau lagi rame yang request
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Fungsi helper buat ngambil session DB tiap ada request API
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()