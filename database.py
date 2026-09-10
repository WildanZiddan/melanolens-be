import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DEFAULT_SUPABASE_URL = "postgresql://postgres.mfxgrtogcmmxpggtvalk:Melanolens_MuhammadRidha@aws-1-ap-southeast-1.pooler.supabase.com:5432/postgres?sslmode=require"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_SUPABASE_URL)

try:
    if DATABASE_URL.startswith("sqlite"):
        engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    else:
        temp_engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True, 
            pool_recycle=1800,     
            connect_args={"connect_timeout": 3}
        )
        with temp_engine.connect() as conn:
            pass
        engine = temp_engine
except Exception as err:
    print(f"[Database] Remote DB unreachable, falling back to local SQLite: {err}")
    DATABASE_URL = "sqlite:///./melanolens.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
