from fastapi import APIRouter, File, UploadFile, Depends
from dotenv import dotenv_values
import os
import fitz
from docx import Document as DocxDocument
from fastapi.responses import JSONResponse
from .utils import verify_token

config = dotenv_values(".env")

router = APIRouter(prefix="/document", tags=["document"])

DOCUMENTS_FOLDER = os.path.join(os.path.dirname(__file__), "..", "documents")

if not os.path.exists(DOCUMENTS_FOLDER):
    os.makedirs(DOCUMENTS_FOLDER)


async def convert_pdf_to_md(file_path):
    doc = fitz.open(file_path)
    markdown_content = ""

    for page in doc:
        markdown_content += page.get_text() + "\n"

    return markdown_content


async def convert_docx_to_md(file_path):
    doc = DocxDocument(file_path)
    markdown_content = ""

    for para in doc.paragraphs:
        markdown_content += para.text + "\n"

    return markdown_content


async def convert_to_md(file_path):
    if file_path.endswith('.pdf'):
        return await convert_pdf_to_md(file_path)
    elif file_path.endswith('.docx'):
        return await convert_docx_to_md(file_path)
    else:
        raise ValueError("Unsupported file type")


@router.post("/upload")
async def upload_document(token: str =  Depends(verify_token), document: UploadFile = File(...)):
    content = await document.read()
    
    file_path = os.path.join(DOCUMENTS_FOLDER, document.filename)

    with open(file_path, "wb") as f:
        f.write(content)

    try:
      markdown_content = await convert_to_md(file_path)

      # Save the Markdown content to a .md file
      md_file_path = os.path.splitext(file_path)[0] + ".md"
      with open(md_file_path, "w", encoding="utf-8") as md_file:
            md_file.write(markdown_content)

      return JSONResponse(content={
            "filename": document.filename,
            "filepath": file_path,
            "markdown_filepath": md_file_path,
            "message": "File converted successfully."
      })
    
    except Exception as e:
      return JSONResponse(content={"error": str(e)}, status_code=400)