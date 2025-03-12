"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Optional, List
from pathlib import Path

from langchain_core.runnables import RunnableConfig, ensure_config


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-3-5-sonnet-20240620",
        metadata={
            "description": "The name of the language model to use"
        },
    )

    docs_path: Path = field(
        default=Path("docs"),
        metadata={
            "description": "Path to documentation files for RAG"
        },
    )

    component_output_dir: Path = field(
        default=Path("generated_components"),
        metadata={
            "description": "Directory where generated React components will be saved"
        },
    )

    dev_server_port: int = field(
        default=3000,
        metadata={
            "description": "Port number for the development server"
        },
    )

    screenshot_sizes: List[dict] = field(
        default_factory=lambda: [
            {"width": 1920, "height": 1080, "name": "desktop"},
            {"width": 375, "height": 667, "name": "mobile"}
        ],
        metadata={
            "description": "Screen sizes for taking screenshots"
        },
    )

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
