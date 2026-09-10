import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DEFAULT_SUPABASE_URL = "postgresql://postgres.mfxgrtogcmmxpggtvalk:Melanolens_MuhammadRidha@aws-1-ap-southeast-1.pooler.supabase.com:5432/postgres?sslmode=require"
ENV_DATABASE_URL = os.getenv("DATABASE_URL")

engine = None

# Attempt 1: Try ENV_DATABASE_URL if present and not placeholder
if ENV_DATABASE_URL and "[YOUR-PASSWORD]" not in ENV_DATABASE_URL:
    try:
        print("[Database] Trying environment DATABASE_URL...")
        if ENV_DATABASE_URL.startswith("sqlite"):
            engine = create_engine(ENV_DATABASE_URL, connect_args={"check_same_thread": False})
        else:
            temp_engine = create_engine(
                ENV_DATABASE_URL,
                pool_pre_ping=True,
                pool_recycle=1800,
                connect_args={"connect_timeout": 5}
            )
            with temp_engine.connect() as conn:
                pass
            engine = temp_engine
            print("[Database] Successfully connected using environment DATABASE_URL!")
    except Exception as err:
        print(f"[Database] Environment DATABASE_URL failed: {err}")
        engine = None

# Attempt 2: Try DEFAULT_SUPABASE_URL if Attempt 1 failed or wasn't tried
if engine is None:
    try:
        print("[Database] Trying DEFAULT_SUPABASE_URL...")
        temp_engine = create_engine(
            DEFAULT_SUPABASE_URL,
            pool_pre_ping=True,
            pool_recycle=1800,
            connect_args={"connect_timeout": 5}
        )
        with temp_engine.connect() as conn:
            pass
        engine = temp_engine
        print("[Database] Successfully connected to default Supabase database!")
    except Exception as err:
        print(f"[Database] Default Supabase connection failed: {err}")
        engine = None

# Attempt 3: Fallback to local SQLite if both remote connections failed
if engine is None:
    print("[Database] All remote DB connections failed, falling back to local SQLite: melanolens.db")
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
