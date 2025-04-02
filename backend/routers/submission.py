import asyncio
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic_models import submission_schemas
from database import submission_collection
from bson import ObjectId
from .utils import verify_token
from .logger import logger

router = APIRouter(prefix="/submission", tags=["submission"])

# ---------------------------
# Helper: Process one document concurrently
# ---------------------------
async def process_document_logic(document_id: str):
    """
    Processes a single document by calling the heavy processing logic.
    Assumes that process_document (from routers/document.py) is asynchronous.
    """
    # Use a relative import based on your project structure.
    from .document import process_document
    await process_document(document_id)

# ---------------------------
# Helper: Process multiple documents concurrently using asyncio.gather
# ---------------------------
async def process_documents_background(document_ids: list):
    tasks = [process_document_logic(doc_id) for doc_id in document_ids]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Error processing document {document_ids[i]}: {result}")

# ---------------------------
# Final Submission Endpoint (POST /submission/submit)
# ---------------------------
@router.post("/submit")
async def submit_manuscript(
    submission: submission_schemas.SubmissionCreate,
    token_data: dict = Depends(verify_token),
    background_tasks: BackgroundTasks = None
):
    """
    Final manuscript submission.
    Creates a submission record immediately and schedules background processing
    for each document concurrently using asyncio.gather.
    Returns immediately so that the user can be redirected to a common summary page.
    """
    logger.info("Processing final manuscript submission")
    try:
        user_id = token_data.get("user_id")
        submission_data = {
            "title": submission.title,
            "abstract": submission.abstract,
            "keywords": submission.keywords,
            "authors": [author.dict() for author in submission.authors],
            "document_ids": submission.document_ids,  # Array of document IDs from frontend
            "reviewers": [rev.dict() for rev in submission.reviewers],
            "letter": submission.letter,
            "submission_date": datetime.now(timezone.utc),
            "user_id": ObjectId(user_id)
        }
        new_submission = await submission_collection.insert_one(submission_data)
        submission_id = str(new_submission.inserted_id)
        
        # Schedule background processing for all documents concurrently.
        if background_tasks:
            background_tasks.add_task(process_documents_background, submission_data["document_ids"])
        
        logger.info("Manuscript submitted successfully")
        return {"message": "Manuscript submitted successfully", "submission_id": submission_id}
    
    except Exception as e:
        logger.error(f"Error during final submission: {str(e)}")
        return JSONResponse(content={"error": f"Submission failed: {str(e)}"}, status_code=500)

# ---------------------------
# GET Endpoint to Retrieve All Submissions (GET /submission/)
# ---------------------------
@router.get("/")
async def get_all_submissions(token_data: dict = Depends(verify_token)):
    """
    Retrieve all submissions for the authenticated user.
    """
    try:
        user_id = token_data.get("user_id")
        submissions = []
        cursor = submission_collection.find({"user_id": ObjectId(user_id)})
        async for sub in cursor:
            submissions.append(submission_schemas.SubmissionResponse(**sub))
        return {"submissions": submissions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving submissions: {str(e)}")

# ---------------------------
# GET Endpoint to Retrieve a Specific Submission (GET /submission/{submission_id})
# ---------------------------
@router.get("/{submission_id}")
async def get_submission(submission_id: str, token_data: dict = Depends(verify_token)):
    """
    Retrieve a specific submission by ID.
    """
    submission = await submission_collection.find_one({"_id": ObjectId(submission_id)})
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    return submission_schemas.SubmissionResponse(**submission)
