from __future__ import annotations
import base64, hashlib, hmac, json, os, time
from dotenv import load_dotenv
load_dotenv()
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

router=APIRouter(prefix="/api/v1/auth",tags=["Authentication"])
bearer=HTTPBearer(auto_error=False)
SECRET=os.getenv("JWT_SECRET","")
USERS={"admin":{"password":os.getenv("ADMIN_PASSWORD",""),"role":"admin"},
       "analyst":{"password":os.getenv("ANALYST_PASSWORD",""),"role":"analyst"},
       "viewer":{"password":os.getenv("VIEWER_PASSWORD",""),"role":"viewer"}}

class Login(BaseModel):
    username:str
    password:str

def _b64(data:bytes)->str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

def issue_token(username:str,role:str)->str:
    if not SECRET: raise HTTPException(status_code=503,detail="JWT_SECRET is not configured.")
    now=int(time.time()); head=_b64(b'{"alg":"HS256","typ":"JWT"}')
    body=_b64(json.dumps({"sub":username,"role":role,"iat":now,"exp":now+3600}).encode())
    message=f"{head}.{body}"
    return f"{message}.{_b64(hmac.new(SECRET.encode(),message.encode(),hashlib.sha256).digest())}"

def current_user(credentials:HTTPAuthorizationCredentials|None=Depends(bearer))->dict:
    if not SECRET or credentials is None: raise HTTPException(status_code=401,detail="Bearer token required.")
    try:
        head,body,signature=credentials.credentials.split("."); message=f"{head}.{body}"
        expected=_b64(hmac.new(SECRET.encode(),message.encode(),hashlib.sha256).digest())
        if not hmac.compare_digest(signature,expected): raise ValueError("signature")
        payload=json.loads(base64.urlsafe_b64decode(body+"="*(-len(body)%4)))
        if payload["exp"]<time.time(): raise ValueError("expired")
        return payload
    except Exception as exc: raise HTTPException(status_code=401,detail="Invalid or expired token.") from exc

def require_roles(*roles:str):
    def check(user:dict=Depends(current_user)):
        if user.get("role") not in roles: raise HTTPException(status_code=403,detail="Insufficient role.")
        return user
    return check

@router.post("/token")
def login(data:Login):
    user=USERS.get(data.username)
    if not user or not user["password"] or not hmac.compare_digest(data.password,user["password"]):
        raise HTTPException(status_code=401,detail="Invalid credentials.")
    return {"access_token":issue_token(data.username,user["role"]),"token_type":"bearer","expires_in":3600}
