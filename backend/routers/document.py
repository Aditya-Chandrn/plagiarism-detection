from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from dotenv import dotenv_values
import os
from fastapi.responses import JSONResponse, FileResponse
from .utils import verify_token, convert_to_md
import random
from concurrent.futures import ProcessPoolExecutor
import asyncio
import shutil
import aiofiles

config = dotenv_values(".env")

router = APIRouter(prefix="/document", tags=["document"])

DOCUMENTS_FOLDER = os.path.join(os.path.dirname(__file__), "..", "documents")

# Ensure the documents folder exists
os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

# Create a global executor for CPU-bound tasks
executor = ProcessPoolExecutor()

def detect_similarity() -> float:
    for i in range(1, 100):
        # print(f"i: {i}")
        pass
    
    return random.random()

def detect_ai_generated_content() -> float:
    for j in range(1, 100):
        # print(f"j: {j}")
        pass

    return random.random()

@router.post("/upload")
async def upload_document(token: str = Depends(verify_token), document: UploadFile = File(...)):
    """
    Endpoint to upload a document, convert it to Markdown, and compute AI-generated content and text similarity scores.
    """
    file_path = os.path.join(DOCUMENTS_FOLDER, document.filename)

    # Save the uploaded file to the server
    try:
        content = await document.read()
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)

    except Exception as e:
        return JSONResponse(content={"error": f"Failed to save file: {str(e)}"}, status_code=500)

    # Convert the file to Markdown
    try:
        md_file_path = await convert_to_md(file_path)

        # Move the markdown file to a standardized location if needed
        standardized_md_path = os.path.join(DOCUMENTS_FOLDER, os.path.basename(md_file_path))
        shutil.move(md_file_path, standardized_md_path)
    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": f"Markdown conversion failed: {str(e)}"}, status_code=500)

    # Run CPU-bound tasks using ProcessPoolExecutor
    try:
        ai_score_future = executor.submit(detect_ai_generated_content)
        text_similarity_future = executor.submit(detect_similarity)

        # Await the results asynchronously
        ai_score_ans, text_similarity_ans = await asyncio.gather(
            asyncio.to_thread(ai_score_future.result),
            asyncio.to_thread(text_similarity_future.result)
        )
    except Exception as e:
        return JSONResponse(content={"error": f"Score computation failed: {str(e)}"}, status_code=500)

    return JSONResponse(content={
        "filename": document.filename,
        "filepath": file_path,
        "markdown_filepath": standardized_md_path,
        "message": "File converted successfully.",
        "ai_gen_score": ai_score_ans,
        "text_similarity_score": text_similarity_ans
    })

@router.get("/{filename}")
async def get_document(filename: str):    
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DOCUMENTS_DIR = os.path.join(ROOT_DIR, "documents")

    file_path = os.path.join(DOCUMENTS_DIR, filename)


    print(f"Here {file_path}")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="File not found")
