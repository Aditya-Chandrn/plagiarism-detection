from typing_extensions import Annotated, Literal, Union
from pydantic import Field, TypeAdapter, BaseModel


class ResearchPaper(BaseModel):
      document_type: Literal["research_paper"] = "research_paper";
      pass

class Assignment(BaseModel):
      document_type: Literal["assignemnt"] = "assignment";
      pass


DocumentUnion = Annotated[Union[ResearchPaper, Assignment], Field(discriminator="document_type")]
DocumentUnionType = TypeAdapter(DocumentUnion)

class Document(BaseModel):
      pass


class DocumentResponse(BaseModel):
      pass