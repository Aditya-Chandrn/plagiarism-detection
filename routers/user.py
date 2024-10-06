import datetime
from typing import Any, Dict
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from database import user_collection
from pydantic_models import user_schema
import bcrypt
from jose import JWTError, jwt
from jose.exceptions import ExpiredSignatureError
from dotenv import dotenv_values

config = dotenv_values(".env")

router = APIRouter(prefix="/users", tags=["users"])


def create_access_token(data: Dict[str, Any]) -> str:
    encoded_jwt = jwt.encode(data, str(config["SECRET_KEY"]), algorithm=config["ALGORITHM"])
    return encoded_jwt



@router.post("/signup", response_model=user_schema.User, status_code=status.HTTP_201_CREATED)
async def register_user(user: user_schema.UserBase): 

      existing_user = await user_collection.find_one(
            {"email": user.email}
      )

      if existing_user is not None:
            return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST, content={"detail": "Email is already registered."}
      )


      hashed_password = bcrypt.hashpw(password=user.password.encode("utf-8"), salt=bcrypt.gensalt()).decode("utf-8")
      user.password = hashed_password
      new_user = await user_collection.insert_one(user.model_dump(by_alias=True))

      db_user = await user_collection.find_one(
            {"_id": new_user.inserted_id}
      )
      return db_user
      

@router.post("/login", response_model=user_schema.User, status_code=status.HTTP_200_OK)
async def login_user(user: user_schema.UserLogin):
      db_user = await user_collection.find_one({"email": user.email})

      if db_user is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST, 
            content={"detail": "User with the given email doesn't exist."}
        )

      
      if not bcrypt.checkpw(user.password.encode("utf-8"), db_user['password'].encode("utf-8")):
            return JSONResponse(
                  status_code=status.HTTP_400_BAD_REQUEST, 
                  content={"detail": "Incorrect Password"}
            )
    
      # ------- CREATE AN ACCESS TOKEN  -------
      return db_user
