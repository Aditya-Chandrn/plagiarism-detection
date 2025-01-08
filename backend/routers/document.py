from datetime import datetime, timezone
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from dotenv import dotenv_values
import os
from fastapi.responses import JSONResponse, FileResponse
from .utils import verify_token, convert_to_md, detect_ai_generated_content, detect_similarity, read_md_file, scrape_and_save_research_papers
from concurrent.futures import ThreadPoolExecutor
import shutil
from pydantic_models import document_schemas
from database import document_collection
from bson import ObjectId
from .logger import logger

config = dotenv_values(".env")

router = APIRouter(prefix="/document", tags=["document"])

DOCUMENTS_FOLDER = os.path.join(os.path.dirname(__file__), "..", "documents")

# Ensure the documents folder exists
os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

# Create a global executor for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=5)


@router.post("/upload")
async def upload_document(token_data: dict = Depends(verify_token), document: UploadFile = File(...)):
    logger.info("Starting document upload process")
    
    # Ensure DOCUMENTS_FOLDER is defined and used correctly
    if not os.path.exists(DOCUMENTS_FOLDER):
        logger.info("Creating documents folder")
        os.makedirs(DOCUMENTS_FOLDER)

    # Save the uploaded document to a standard location
    try:
        file_path = os.path.join(DOCUMENTS_FOLDER, document.filename)
        logger.info(f"Saving file...")
        # Save file
        content = await document.read()
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"File saved successfully")

    except Exception as e:
        logger.error(f"Error while saving file: {str(e)}")
        return JSONResponse(content={"error": f"Failed to save file: {str(e)}"}, status_code=500)

    # Convert to Markdown
    try:
        logger.info(f"Converting file to markdown...")
        md_file_path = await convert_to_md(file_path)
        logger.info(f"Converted file to markdown")

        # Move Markdown to standardized location
        standardized_md_path = os.path.join(
            DOCUMENTS_FOLDER, os.path.basename(md_file_path))
        
        if os.path.exists(standardized_md_path):
            logger.info(f"Removing existing markdown file...")
            os.remove(standardized_md_path)  # Remove the existing file if it exists
        
        shutil.move(md_file_path, standardized_md_path)
        logger.info(f"Markdown file moved to standardized location")

    except ValueError as e:
        logger.error(f"Markdown conversion error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=400)
    except Exception as e:
        logger.error(f"Markdown conversion failed: {str(e)}")
        return JSONResponse(content={"error": f"Markdown conversion failed: {str(e)}"}, status_code=500)

    # Read MD Content and Extract Title
    try:
        logger.info(f"Reading markdown file...")
        md_content = read_md_file(standardized_md_path)
        title = md_content.split('\n')[0].replace('#', '').strip()
        logger.info(f"Extracted title from markdown")

        # Scrape papers from ArXiv 
        logger.info(f"Scraping papers from ArXiv...")
        scraped_paper_details = await scrape_and_save_research_papers(title)

        for paper in scraped_paper_details:
            logger.info(f"Converting scraped paper to markdown")
            path = await convert_to_md(paper['path'])
            paper['md_path'] = path
        
        logger.info(f"Scraped papers converted to markdown")
       
    except Exception as e:
        logger.error(f"Error while scraping papers: {str(e)}")
        return JSONResponse(content={"error": f"Failed to scrape papers: {str(e)}"}, status_code=500)


    # AI and Similarity Score Computation
    results = []

    try:
        logger.info(f"Starting AI and similarity score computation")
        ai_score_future = executor.submit(
            detect_ai_generated_content, standardized_md_path)
        text_similarity_futures = [
            executor.submit(detect_similarity, standardized_md_path, paper['md_path'], paper)
            for paper in scraped_paper_details
        ]
    
        # Wait for all futures to complete
        ai_score = ai_score_future.result()
        text_similarity_scores = [future.result() for future in text_similarity_futures]
    
        results.append({
            "ai_score": ai_score,
            "text_similarity_scores": text_similarity_scores
        })
        logger.info(f"AI and similarity score computation completed")

    except Exception as e:
        logger.error(f"Error during AI and similarity score computation: {str(e)}")
        return JSONResponse(content={"error": f"Failed to compute AI and similarity scores: {str(e)}"}, status_code=500)
    
    # Create DB Record for the uploaded document
    try:
        logger.info("Creating DB record for the uploaded document")
        ai_score_dict = [ai.dict() for ai in ai_score]
        similarity_score_dict = text_similarity_scores
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

        logger.info(f"Document uploaded and saved to DB successfully")
        return document_schemas.Document.model_validate(db_document)
    
    except Exception as e:
        logger.error(f"Error while creating DB record: {str(e)}")
        return JSONResponse(content={"error": f"Failed to create DB record: {str(e)}"}, status_code=500)

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
