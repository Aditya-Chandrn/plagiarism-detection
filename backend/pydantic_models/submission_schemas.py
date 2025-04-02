from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from typing_extensions import Annotated
from pydantic.functional_validators import BeforeValidator

# Use the same approach as your document schema for ObjectId conversion.
PyObjectId = Annotated[str, BeforeValidator(str)]

class Author(BaseModel):
    name: str
    email: str
    type: str  # e.g., "Author", "Co-Author", "Mentor", etc.

    model_config = ConfigDict(from_attributes=True)

class Reviewer(BaseModel):
    name: str
    email: str

    model_config = ConfigDict(from_attributes=True)

class SubmissionCreate(BaseModel):
    title: str
    abstract: str
    authors: List[Author]
    keywords: List[str]
    reviewers: List[Reviewer]
    letter: Optional[str] = None
    # References to documents stored in the documents collection (as strings)
    document_ids: Optional[List[PyObjectId]] = None

    model_config = ConfigDict(from_attributes=True)

class SubmissionResponse(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    title: str
    abstract: str
    authors: List[Author]
    keywords: List[str]
    reviewers: List[Reviewer]
    letter: Optional[str] = None
    submission_date: datetime
    user_id: PyObjectId
    document_ids: Optional[List[PyObjectId]] = None

    model_config = ConfigDict(from_attributes=True)
