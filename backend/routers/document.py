from fastapi import APIRouter, File, UploadFile, Depends
from dotenv import dotenv_values
import os
from fastapi.responses import JSONResponse
from .utils import verify_token, convert_to_md
import random
from concurrent.futures import ProcessPoolExecutor
import asyncio

config = dotenv_values(".env")

router = APIRouter(prefix="/document", tags=["document"])

DOCUMENTS_FOLDER = os.path.join(os.path.dirname(__file__), "..", "documents")

# Ensure the documents folder exists
os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

# Create a global executor for CPU-bound tasks
executor = ProcessPoolExecutor()


def detect_similarity() -> float:
    for i in range(1, 100):
        print(f"i: {i}")
    
    return random.random()


def detect_ai_generated_content() -> float:
    for j in range(1, 100):
        print(f"j: {j}")

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
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        return JSONResponse(content={"error": f"Failed to save file: {str(e)}"}, status_code=500)

    # Convert the file to Markdown
    try:
        markdown_content = await convert_to_md(file_path)

        # Save the converted Markdown to a .md file
        md_file_path = os.path.splitext(file_path)[0] + ".md"
        with open(md_file_path, "w", encoding="utf-8") as md_file:
            md_file.write(markdown_content)
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
        "markdown_filepath": md_file_path,
        "message": "File converted successfully.",
        "ai_gen_score": ai_score_ans,
        "text_similarity_score": text_similarity_ans
    })
