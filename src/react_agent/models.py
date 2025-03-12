from pydantic import BaseModel, Field
from typing import List


class FileItem:
    """Content of one file in the extracted code."""
    filename: str
    content: str
    file_type: str
    entrypoint: bool = False


class FileContents:
    """Represents the complete structure returned by the extraction step."""
    dependencies: List[str] = Field(default_factory=list)
    items: List[FileItem] = Field(default_factory=list)

class Chunk(BaseModel):
    """Structured output for vector chunks."""
    content: str = Field(
        description="Content of the chunk"
    )
    package_name: str = Field(
        description="Package name of the chunk (e.g. @octagon/analytics). If the chunk is not part of a package, this will be empty"
    )


class VectorChunks(BaseModel):
    """Structured output for vector chunks."""
    chunks: List[Chunk] = Field(
        description="List of chunks for use in the vector database"
    )

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

class ResearchQuestion(BaseModel):
    """Structured output for queries."""
    question: str = Field(
        description="Specific query to run against doc resources"
    )
    filter: str = Field(
        description="Filter to narrow down the results for better accuracy"
    )

class ResearchQuestions(BaseModel):
    """Structured output for design analysis and building queries."""
    design_analysis: List[str] = Field(
        description="Key observations about the design"
    )
    questions: List[ResearchQuestion] = Field(
        description="Specific queries to run against doc resources"
    )

class DependencyList(BaseModel):
    """Structured output for dependency list."""
    deps: List[str] = Field(
        description="List of dependencies"
    )