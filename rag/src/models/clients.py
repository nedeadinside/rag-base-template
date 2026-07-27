from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    """
    A single chunk returned by docling-serve.
    """

    model_config = ConfigDict(extra="ignore")

    text: str
    headings: list[str] = Field(default_factory=list)
