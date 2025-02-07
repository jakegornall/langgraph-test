from pydantic import BaseModel, Field
from typing import List, Dict, Any
from dataclasses import field


class SearchResult(BaseModel):
    """Model for documentation search results."""
    content: str = Field(description="Relevant doc content")
    source: str = Field(description="Documentation source reference")
    score: float = Field(description="Relevance score")


class FileItem:
    """Content of one file in the extracted code."""
    filename: str
    content: str
    file_type: str
    entrypoint: bool = False


class FileContents:
    """Represents the complete structure returned by the extraction step."""
    dependencies: List[str] = field(default_factory=list)
    items: List[FileItem] = field(default_factory=list)



class ValidationResult(BaseModel):
    """Structured output for design validation."""
    passed: bool = Field(
        description="Whether the implementation matches the design"
    )
    discrepancies: List[str] = Field(
        description="List of discrepancies or issues found"
    )
    matches: List[str] = Field(
        description="List of elements that match the design perfectly"
    )


class ResearchQuestions(BaseModel):
    """Structured output for design analysis and building queries."""
    design_analysis: List[str] = Field(
        description="Key observations about the design"
    )
    questions: List[str] = Field(
        description="Specific queries to run against doc resources"
    )