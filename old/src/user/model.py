from pydantic import EmailStr, Field, BeforeValidator
from beanie import Document
from typing import List
from datetime import datetime
from typing_extensions import Annotated

PyObjectId = Annotated[str, BeforeValidator(str)]

# User Model
class User(Document):
  email: EmailStr
  password: str
  fname: str
  lname: str
  affiliation: str
  documents: List[str] = Field(default_factory=list)
  timestamp: datetime = Field(default_factory=datetime.utcnow)
  
  class Settings:
    collection = "users"
    
  class Config: 
    arbitrary_types_allowed = True