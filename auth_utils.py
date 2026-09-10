import jwt
import datetime
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = "Melanolens-MuhammadRidha"
ALGORITHM = "HS256"

def hash_password(password: str):
    return pwd_context.hash(password[:72])

def verify_password(plain_password: str, hashed_password: str):
    try:
        if not hashed_password or not isinstance(hashed_password, str):
            return False
        return pwd_context.verify(plain_password[:72], hashed_password)
    except Exception as e:
        print(f"[Verify Password Error]: {e}")
        return False

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(days=1)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
