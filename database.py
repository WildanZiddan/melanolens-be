import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Primary working Supabase connection using Session/Transaction Pooler (Port 6543 & 5432)
SUPABASE_URL_6543 = "postgresql://postgres.mfxgrtogcmmxpggtvalk:Melanolens_MuhammadRidha@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres?sslmode=require"
SUPABASE_URL_5432 = "postgresql://postgres.mfxgrtogcmmxpggtvalk:Melanolens_MuhammadRidha@aws-1-ap-southeast-1.pooler.supabase.com:5432/postgres?sslmode=require"
ENV_DATABASE_URL = os.getenv("DATABASE_URL")

engine = None

# Attempt 1: Try ENV_DATABASE_URL if valid postgresql and no placeholder
if ENV_DATABASE_URL and ENV_DATABASE_URL.startswith("postgresql") and "[YOUR-PASSWORD]" not in ENV_DATABASE_URL and "db.mfxgrtogcmmxpggtvalk.supabase.co" not in ENV_DATABASE_URL:
    try:
        print("[Database] Trying ENV_DATABASE_URL...")
        temp_engine = create_engine(ENV_DATABASE_URL, pool_pre_ping=True, pool_recycle=1800, connect_args={"connect_timeout": 5})
        with temp_engine.connect() as conn:
            pass
        engine = temp_engine
        print("[Database] Successfully connected using ENV_DATABASE_URL!")
    except Exception as err:
        print(f"[Database] ENV_DATABASE_URL failed: {err}")
        engine = None

# Attempt 2: Try SUPABASE_URL_6543 (Pooler - IPv4 compatible)
if engine is None:
    try:
        print("[Database] Trying SUPABASE_URL_6543 (Pooler)...")
        temp_engine = create_engine(SUPABASE_URL_6543, pool_pre_ping=True, pool_recycle=1800, connect_args={"connect_timeout": 5})
        with temp_engine.connect() as conn:
            pass
        engine = temp_engine
        print("[Database] Successfully connected to SUPABASE_URL_6543!")
    except Exception as err:
        print(f"[Database] SUPABASE_URL_6543 failed: {err}")
        engine = None

# Attempt 3: Try SUPABASE_URL_5432
if engine is None:
    try:
        print("[Database] Trying SUPABASE_URL_5432...")
        temp_engine = create_engine(SUPABASE_URL_5432, pool_pre_ping=True, pool_recycle=1800, connect_args={"connect_timeout": 5})
        with temp_engine.connect() as conn:
            pass
        engine = temp_engine
        print("[Database] Successfully connected to SUPABASE_URL_5432!")
    except Exception as err:
        print(f"[Database] SUPABASE_URL_5432 failed: {err}")
        engine = None

# Attempt 4: Fallback to local SQLite only if all remote connections fail
if engine is None:
    print("[Database] WARNING: All Supabase connections failed, falling back to local SQLite!")
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
