from pydantic import ConfigDict, BaseModel, Field, EmailStr
from typing_extensions import Annotated, Optional
from pydantic.functional_validators import BeforeValidator
from pydantic_models import user_schema
PyObjectId = Annotated[str, BeforeValidator(str)]


class UserBase(BaseModel):
      name: str 
      email: str
      password: str


class UserLogin(BaseModel):
      email: str
      password: str


class User(BaseModel):
      id: Optional[PyObjectId] = Field(alias="_id", default=None)
      name: str 
      email: str
      password: str

      model_config = ConfigDict()
      model_config["from_attributes"] = True