from pydantic import BaseModel, Field, BeforeValidator
from beanie import Document
from typing import List
from datetime import datetime
from typing_extensions import Annotated

# Plagiarism Model
class PlagiarisedContent(BaseModel):
  line: str
  reference: str

class Plagiarism(BaseModel):
  score: float
  lines: List[PlagiarisedContent]
  
# AI Content Model
class AIContent(BaseModel):
  score: float
  lines: List[str]

PyObjectId = Annotated[str, BeforeValidator(str)]

# Document Model
class Doc(Document):
  file: str
  userId: str
  plagiarism: Plagiarism
  ai_content: AIContent
  timestamp: datetime = Field(default_factory=datetime.utcnow)
  
  class Settings:
    collection = "docs"
    
  class Config: 
    arbitrary_types_allowed = True