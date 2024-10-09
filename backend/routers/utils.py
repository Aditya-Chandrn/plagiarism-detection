from fastapi import HTTPException, Request, status
from jose import JWTError, jwt
from jose.exceptions import ExpiredSignatureError
from dotenv import dotenv_values


config = dotenv_values(".env")

async def verify_token(request: Request):
      token = request.headers.get("Authorization")
      if token: 
            token = token.replace("Bearer ", "")
            try:
                  jwt.decode(token, config["SECRET_KEY"], algorithms=[config["ALGORITHM"]])
            except ExpiredSignatureError:
                  raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Token has expired",
                        headers={"WWW-Authenticate": "Bearer"},
                  )
            except JWTError:
                  raise HTTPException(
                  status_code=status.HTTP_401_UNAUTHORIZED,
                  detail="Could not validate credentials",
                  headers={"WWW-Authenticate": "Bearer"},
                  )
      else:
            raise HTTPException(
                  status_code=status.HTTP_401_UNAUTHORIZED,
                  detail="Unauthorized",
                  headers={"WWW-Authenticate": "Bearer"},
            )
      return token