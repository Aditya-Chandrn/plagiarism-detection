from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr

from utils.print_log import log_success, log_error


router = APIRouter()

# Get Doc Scores
class NewDoc(BaseModel):
  email: EmailStr
  
@router.post("/get-scores")
async def register(new_doc: NewDoc):
  log_success(new_doc)
  return {"message": "Doc Scores: 1 2"}




