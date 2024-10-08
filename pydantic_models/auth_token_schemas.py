from pydantic import ConfigDict, BaseModel, Field, EmailStr
from typing_extensions import Annotated, Optional
from pydantic.functional_validators import BeforeValidator
from pydantic_models import user_schema
PyObjectId = Annotated[str, BeforeValidator(str)]


class AuthToken(BaseModel):
      auth_token: str

      model_config = ConfigDict()
      model_config["from_attributes"] = True