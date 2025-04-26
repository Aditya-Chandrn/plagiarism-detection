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
    background_tasks: BackgroundTasks = None  # you can even drop this param if you like
):
    logger.info("Processing final manuscript submission")
    try:
        user_id = token_data.get("user_id")
        submission_data = {
            "title": submission.title,
            "abstract": submission.abstract,
            "keywords": submission.keywords,
            "authors": [author.dict() for author in submission.authors],
            "document_ids": submission.document_ids,
            "reviewers": [rev.dict() for rev in submission.reviewers],
            "letter": submission.letter,
            "submission_date": datetime.now(timezone.utc),
            "user_id": ObjectId(user_id)
        }
        new_submission = await submission_collection.insert_one(submission_data)
        submission_id = str(new_submission.inserted_id)

        logger.info("Manuscript submitted successfully")
        return {
            "message": "Manuscript submitted successfully",
            "submission_id": submission_id
        }
    except Exception as e:
        logger.error(f"Error during final submission: {str(e)}")
        return JSONResponse(
            content={"error": f"Submission failed: {str(e)}"},
            status_code=500
        )

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
# GET Endpoint to Retrieve Submission details by document id
# ---------------------------


@router.get("/by-document/{document_id}", response_model=submission_schemas.SubmissionResponse)
async def get_submission_by_document(
    document_id: str,
    token_data: dict = Depends(verify_token),
):
    """
    Return the submission that contains the given document_id in its document_ids.
    """
    submission = await submission_collection.find_one({
        "document_ids": document_id  # match on string ID
    })
    if not submission:
        raise HTTPException(404, "Submission not found for this document")

    return submission_schemas.SubmissionResponse(**submission)


# ---------------------------
# GET Endpoint to Retrieve a Specific Submission (GET /submission/{submission_id})
# ---------------------------

@router.get("/{submission_id}", response_model=submission_schemas.SubmissionResponse)
async def get_submission(
    submission_id: str,
    token_data: dict = Depends(verify_token),
):
    submission = await submission_collection.find_one({"_id": ObjectId(submission_id)})
    if not submission:
        raise HTTPException(404, "Submission not found")
    return submission_schemas.SubmissionResponse(**submission)
