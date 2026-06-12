import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, Text
from sqlalchemy.orm import relationship
from database import Base

class MelMsUser(Base):
    __tablename__ = "mel_msuser"

    user_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_nama = Column(String(100), nullable=False)
    user_email = Column(String(100), unique=True, index=True, nullable=False)
    user_password = Column(String(255), nullable=False)
    user_tanggalLahir = Column(DateTime, nullable=True)
    user_jenisKelamin = Column(String(20), nullable=True)
    user_pekerjaan = Column(String(100), nullable=True)
    user_role = Column(String(20), default="user")
    user_status = Column(Integer, default=1)
    user_createAt = Column(DateTime, default=datetime.datetime.utcnow)
    user_updateAt = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    scans = relationship("MelTrScan", back_populates="owner")

class MelTrScan(Base):
    __tablename__ = "mel_trscan"

    scan_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("mel_msuser.user_id"), nullable=False)
    scan_gambar = Column(Text, nullable=False)
    scan_tanggal = Column(DateTime, default=datetime.datetime.utcnow)
    scan_persentase = Column(Float, nullable=False)
    scan_respon = Column(Text, nullable=True)
    scan_responGambar = Column(Text, nullable=True)

    owner = relationship("MelMsUser", back_populates="scans")