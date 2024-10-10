from fastapi import HTTPException, Request, status
from jose import JWTError, jwt
from jose.exceptions import ExpiredSignatureError
from dotenv import dotenv_values
import fitz 
from docx import Document as DocxDocument
 
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


async def convert_pdf_to_md(file_path: str) -> str:
    doc = fitz.open(file_path)
    markdown_content = "".join([page.get_text() + "\n" for page in doc])
    return markdown_content


async def convert_docx_to_md(file_path: str) -> str:
    doc = DocxDocument(file_path)
    markdown_content = "\n".join([para.text for para in doc.paragraphs])
    return markdown_content


async def convert_to_md(file_path: str) -> str:
    """
    Convert a file (PDF or DOCX) to Markdown.
    """
    if file_path.endswith('.pdf'):
        return await convert_pdf_to_md(file_path)
    elif file_path.endswith('.docx'):
        return await convert_docx_to_md(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")