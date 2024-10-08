from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from os import getenv

from utils.db import init_db, disconnect_db
from user.router import router as user_router
from doc.router import router as doc_router


load_dotenv()
CLIENT_URL = getenv("CLIENT_URL")

app = FastAPI()
app.add_middleware(
  CORSMiddleware,
  allow_origins=[CLIENT_URL],
  allow_credentials=True,
  allow_methods=["GET", "POST", "DELETE", "PATCH"],
  allow_headers=["*"]
)

@app.on_event("startup")
async def on_startup():
  await init_db()
  
@app.on_event("shutdown")
async def on_shutdown():
  await disconnect_db()
  
# Print blank line after every request
@app.middleware("http")
async def add_line_after_request(request: Request, call_next):
  response = await call_next(request)
  print()
  return response

# Routes
app.include_router(user_router, prefix="/user")
app.include_router(doc_router, prefix="/doc")
