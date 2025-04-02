from datetime import datetime, timedelta
from typing import Any, Dict
from routers.utils import verify_token
from fastapi import APIRouter, status, Request
from fastapi.responses import JSONResponse
from database import user_collection
from pydantic_models import user_schema, auth_token_schemas
import bcrypt
from jose import jwt
from dotenv import dotenv_values

config = dotenv_values(".env")

router = APIRouter(prefix="/user", tags=["user"])


def create_auth_token(data: Dict[str, Any]) -> str:
    data_copy = data.copy()
    expire = datetime.now() + timedelta(minutes=int(config["JWT_EXPIRY"]))
    data_copy.update({"exp": expire})
    token = jwt.encode(data_copy, str(
        config["SECRET_KEY"]), algorithm=config["ALGORITHM"])
    return token


@router.post("/signup", response_model=user_schema.User, status_code=status.HTTP_201_CREATED)
async def register_user(user: user_schema.UserBase):

    existing_user = await user_collection.find_one(
        {"email": user.email}
    )

    if existing_user is not None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST, content={
                "detail": "Email is already registered."}
        )

    hashed_password = bcrypt.hashpw(password=user.password.encode(
        "utf-8"), salt=bcrypt.gensalt()).decode("utf-8")
    user.password = hashed_password
    new_user = await user_collection.insert_one(user.model_dump(by_alias=True))

    db_user = await user_collection.find_one(
        {"_id": new_user.inserted_id}
    )
    return user_schema.User.model_validate(db_user)


@router.post("/login", status_code=status.HTTP_200_OK)
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

    # data = {
    #     "fname": db_user["fname"],
    #     "lname": db_user["lname"],
    #     "email": db_user["email"],
    #     "user_id": str(db_user["_id"]),
    # }
    # auth_token = create_auth_token(data)
    
    display_name = f"{db_user['fname']} {db_user['lname']}"
    data = {
        "display_name": display_name,
        "email": db_user["email"],
        "user_id": str(db_user["_id"]),
    }
    auth_token = create_auth_token(data)

    response = JSONResponse(content={"message": "Login successful"})
    response.set_cookie(
        key="plagiarism-access-token",
        value=auth_token,
        httponly=True,
        secure=True,
        samesite="Strict",
        max_age=3600*24*300     # token expiry in seconds
    )
    return response


@router.post("/logout", status_code=status.HTTP_200_OK)
async def logout_user():
    response = JSONResponse(content={"message": "Logout successful"})
    response.delete_cookie(key="plagiarism-access-token", path="/")
    return response


@router.get("/me")
async def get_current_user(request: Request):
    payload = await verify_token(request)
    display_name = payload.get("display_name", "")
    return {"display_name": display_name}