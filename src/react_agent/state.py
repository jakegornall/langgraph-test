"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Optional, Dict, List, Any
from pathlib import Path

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing import Annotated

@dataclass
class ComponentMetadata:
    """Metadata about a generated React component."""
    name: str
    description: str
    props: Dict[str, str]
    screenshots: Dict[str, Path]
    source_files: Dict[str, Path]

@dataclass
class InputState:
    """Defines the input state for the agent."""
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)
    desktop_design_screenshot: str = field(default="")
    mobile_design_screenshot: str = field(default="")
    requirements: str = field(default="")

@dataclass
class OutputState:
    """Defines the output state containing the generated component."""
    component_zip: Optional[Path] = field(default=None)
    metadata: Optional[ComponentMetadata] = field(default=None)
    success: bool = field(default=False)
    error_message: Optional[str] = field(default=None)

@dataclass
class ValidationState:
    """State related to design validation."""
    passed: bool = field(default=False)
    discrepancies: List[str] = field(default_factory=list)
    matches: List[str] = field(default_factory=list)

@dataclass
class State(InputState):
    """Complete state including component generation progress."""
    is_last_step: IsLastStep = field(default=False)
    
    # Component generation state
    current_component: Optional[ComponentMetadata] = field(default=None)
    generated_code: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    compile_errors: List[str] = field(default_factory=list)
    component_screenshots: Dict[str, str] = field(default_factory=dict)
    
    # Validation state
    validation: Optional[ValidationState] = field(default=None)
    
    # RAG state
    relevant_docs: List[Dict[str, Any]] = field(default_factory=list)
    
    # Extraction state
    extracted: Optional[Dict[str, str]] = field(default=None)
    parse_error: Optional[str] = field(default=None)
    raw_extraction: Optional[Any] = field(default=None)
    
    # Output state
    output: OutputState = field(default_factory=OutputState)
