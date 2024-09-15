from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr

from utils.print_log import log_success, log_error



router = APIRouter()

# Register User
class NewUser(BaseModel):
  email: EmailStr
  fname: str
  lname: str
  affiliation: str
  password: str
  
@router.post("/register")
async def register(new_user: NewUser):
  log_success(new_user)
  return {"message": "User registered successfully"}


# Register User
class LoginCreds(BaseModel):
  email: EmailStr
  password: str
  
@router.post("/login")
async def login(login_creds: LoginCreds):
  log_success(login_creds)
  return {"message": "User authentication successful"}



