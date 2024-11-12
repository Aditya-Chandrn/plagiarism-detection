from datetime import datetime, timezone
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from dotenv import dotenv_values
import os
from fastapi.responses import JSONResponse, FileResponse
from .utils import verify_token, convert_to_md, detect_ai_generated_content, detect_similarity
from concurrent.futures import ProcessPoolExecutor
import asyncio
import shutil
from pydantic_models import document_schemas
from database import document_collection
from bson import ObjectId
from fastapi.encoders import jsonable_encoder
config = dotenv_values(".env")

router = APIRouter(prefix="/document", tags=["document"])

DOCUMENTS_FOLDER = os.path.join(os.path.dirname(__file__), "..", "documents")

# Ensure the documents folder exists
os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

# Create a global executor for CPU-bound tasks
executor = ProcessPoolExecutor()


@router.post("/upload")
async def upload_document(token_data: dict = Depends(verify_token), document: UploadFile = File(...)):
    # Ensure DOCUMENTS_FOLDER is defined and used correctly
    if not os.path.exists(DOCUMENTS_FOLDER):
        os.makedirs(DOCUMENTS_FOLDER)

    # Save the uploaded document to a standard location
    try:
        file_path = os.path.join(DOCUMENTS_FOLDER, document.filename)
        print(f"Saving file to: {file_path}")

        # Save file
        # Save file
        content = await document.read()
        with open(file_path, "wb") as f:
            f.write(content)
        print(f"File saved successfully: {file_path}")

    except Exception as e:
        print(f"Error while saving file: {str(e)}")
        return JSONResponse(content={"error": f"Failed to save file: {str(e)}"}, status_code=500)

    # Convert to Markdown
    # Convert to Markdown
    try:
        md_file_path = await convert_to_md(file_path)
        print(f"Converted to markdown: {md_file_path}")

        # Move Markdown to standardized location
        standardized_md_path = os.path.join(
            DOCUMENTS_FOLDER, os.path.basename(md_file_path))
        shutil.move(md_file_path, standardized_md_path)
        print(f"Markdown file moved to: {standardized_md_path}")

    except ValueError as e:
        print(f"Markdown conversion error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=400)
    except Exception as e:
        print(f"Markdown conversion failed: {str(e)}")
        return JSONResponse(content={"error": f"Markdown conversion failed: {str(e)}"}, status_code=500)

    # AI and Similarity Score Computation
    # AI and Similarity Score Computation
    try:
        ai_score_future = executor.submit(
            detect_ai_generated_content, standardized_md_path)
        text_similarity_future = executor.submit(
            detect_similarity, standardized_md_path)

        # Await the results asynchronously
        ai_score_ans, text_similarity_ans = await asyncio.gather(
            asyncio.to_thread(ai_score_future.result),
            asyncio.to_thread(text_similarity_future.result)
        )
        print(
            f"AI Score: {ai_score_ans}, Text Similarity Score: {text_similarity_ans}")

    except Exception as e:
        print(f"Error in AI/Similarity computation: {str(e)}")
        return JSONResponse(content={"error": f"Score computation failed: {str(e)}"}, status_code=500)

    """
    Create DB Record for the uploaded document.
    """

    ai_score_dict = [ai.dict() for ai in ai_score_ans]
    similarity_score_dict = [sim.dict() for sim in text_similarity_ans]
    user_id = token_data.get("user_id")

    new_document = await document_collection.insert_one({
        "name": document.filename,
        "path": file_path,
        "md_path": standardized_md_path,
        "ai_content_result": ai_score_dict,
        "similarity_result": similarity_score_dict,
        "upload_date": datetime.now(timezone.utc),
        "user_id": ObjectId(user_id)
    })

    db_document = await document_collection.find_one(
        {"_id": new_document.inserted_id}
    )

    return document_schemas.Document.model_validate(db_document)


@router.get("/")
async def get_all_documents(token_data: dict = Depends(verify_token)):
    try:
        user_id = token_data.get("user_id")

        documents = []
        cursor = document_collection.find({"user_id": ObjectId(user_id)})

        async for doc in cursor:
            # doc["_id"] = str(doc["_id"])
            # doc["user_id"] = str(doc["user_id"])

            documents.append(document_schemas.Document(**doc))

        return document_schemas.DocumentResponse(documents=documents)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving documents: {str(e)}"
        )


@router.get("/{document_id}")
async def get_document(document_id: str, token_data: dict = Depends(verify_token)):

    document = await document_collection.find_one({
        "_id": ObjectId(document_id),
    })

    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    document["_id"] = str(document["_id"])
    document["user_id"] = str(document["user_id"])

    return document_schemas.Document(**document)


@router.get("/file/{filename}")
async def get_document(filename: str):
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DOCUMENTS_DIR = os.path.join(ROOT_DIR, "documents")

    file_path = os.path.join(DOCUMENTS_DIR, filename)

    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="File not found")
