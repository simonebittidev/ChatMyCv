from typing import List
from pydantic import BaseModel, Field

class RewrittenQuestion(BaseModel):
    rewritten_question: str = Field(
        description="The clarified and rephrased version of the user's question, rewritten for maximum clarity based on conversation context."
    )

class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities  that " "appear in the text",
    )

class DocumentSummary(BaseModel):
    brief_overview: str = Field(
        description="A brief summary (2-3 sentences) that describes the entire content of the document."
    )
    detailed_summary: str = Field(
        description="A detailed summary covering all information present in the document."
    )

class Chunk(BaseModel):
    title: str = Field(description="A short, descriptive title for this chunk")
    content: str = Field(description="The text content of this chunk")
    keywords: List[str] = Field(description="The keywords extracted from the content of this chunk")

class ChunkedSummary(BaseModel):
    chunks: List[Chunk] = Field(
        description="A list of semantically meaningful chunks, each with a title and content"
    )
