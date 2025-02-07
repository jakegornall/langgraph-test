"""Define a custom React Component Development agent."""

import base64
import shutil
import traceback
import zipfile
from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, START

from react_agent.configuration import Configuration
from react_agent.state import ComponentMetadata, OutputState, State, InputState, ValidationState
from react_agent.tools import compile_component, take_screenshots
from react_agent.utils import extract_code, detect_media_type
from react_agent.prompts import (
    VALIDATION_SYSTEM_PROMPT,
    REACT_DEVELOPER_SYSTEM_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
)
from react_agent.models import ValidationResult, ResearchQuestions
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI


def research(state: State, config: RunnableConfig) -> State:
    """Research needed docs from vector db by analyzing designs and requirements."""
    model = ChatOpenAI(model="gpt-4o")
    structured_llm = model.with_structured_output(ResearchQuestions)
    
    # Create message content for GPT-4V analysis
    content = [
        {
            "type": "text",
            "text": f"""Analyze these design screenshots and requirements to identify key technical aspects that need documentation research:

Requirements:
{state.requirements}"""
        }
    ]
    
    # Add desktop design if available
    if state.desktop_design_screenshot:
        content.extend([
            {"type": "text", "text": "Desktop Design:"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{detect_media_type(state.desktop_design_screenshot)};base64,{state.desktop_design_screenshot}",
                    "detail": "high"
                }
            }
        ])
    
    # Add mobile design if available
    if state.mobile_design_screenshot:
        content.extend([
            {"type": "text", "text": "Mobile Design:"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{detect_media_type(state.mobile_design_screenshot)};base64,{state.mobile_design_screenshot}",
                    "detail": "high"
                }
            }
        ])
    
    # Get structured analysis and questions from GPT-4V
    result = structured_llm.invoke([
        SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
        HumanMessage(content=content)
    ])
    
    # Store the analysis and questions
    return State(
        **{
            **state.__dict__,
            "relevant_docs": [
                {"analysis": result.design_analysis},
                {"questions": result.questions}
            ]
        }
    )

def extract(state: State) -> State:
    """Extract code from the LLM's response."""        
    try:
        extracted = extract_code(state.messages[-1].content)
        return State(
            **{
                **state.__dict__,
                "generated_code": extracted["items"],
                "dependencies": extracted["dependencies"],
                "extracted": extracted,
                "parse_error": None
            }
        )
    except Exception as e:
        return State(
            **{
                **state.__dict__,
                "extracted": extracted,
                "parse_error": str(e)
            }
        )


def compile(state: State, config: RunnableConfig) -> State:
    """Compile the extracted React component via dev server."""
    if not state.generated_code:
        return State(
            **{
                **state.__dict__,
                "compile_errors": ["No code to compile"]
            }
        )

    try:
        result = compile_component(state.generated_code, state.dependencies, config=config)

        if result.get("errors"):
            return State(
                **{
                    **state.__dict__,
                    "compile_errors": result["errors"]
                }
            )

        if result.get("dev_server_url"):
            screenshots = take_screenshots(result["dev_server_url"], config=config)
            return State(
                **{
                    **state.__dict__,
                    "component_screenshots": screenshots,
                    "compile_errors": []
                }
            )

        return state

    except Exception as e:
        return State(
            **{
                **state.__dict__,
                "compile_errors": [str(e)]
            }
        )


def validate(state: State, config: RunnableConfig) -> State:
    """Validate the component's appearance against design screenshots."""
    if not state.component_screenshots or not state.desktop_design_screenshot:
        return State(
            **{
                **state.__dict__,
                "validation": ValidationState(
                    passed=False,
                    discrepancies=["Missing screenshots for validation"]
                )
            }
        )

    model = ChatOpenAI(model="gpt-4o", max_completion_tokens=16384)
    structured_llm = model.with_structured_output(ValidationResult)

    # Prepare comparison images
    content = [
        {
            "type": "text",
            "text": f"Compare these images to validate if the implemented component matches the design:\n\nRequirements:\n{state.requirements}"
        },
        {
            "type": "text",
            "text": "Image 1: Original Desktop Design"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{detect_media_type(state.desktop_design_screenshot)};base64,{state.desktop_design_screenshot}",
                "detail": "high"
            }
        },
        {
            "type": "text",
            "text": "Image 2: Implemented Component (Desktop)"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{detect_media_type(state.component_screenshots['desktop'])};base64,{state.component_screenshots['desktop']}",
                "detail": "high"
            }
        }
    ]

    # Mobile comparison if available
    if state.mobile_design_screenshot and "mobile" in state.component_screenshots:
        content.extend([
            {"type": "text", "text": "Image 3: Original Mobile Design"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{detect_media_type(state.mobile_design_screenshot)};base64,{state.mobile_design_screenshot}",
                    "detail": "high"
                }
            },
            {"type": "text", "text": "Image 4: Implemented Component (Mobile)"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{detect_media_type(state.component_screenshots['mobile'])};base64,{state.component_screenshots['mobile']}",
                    "detail": "high"
                }
            }
        ])

    messages = [
        SystemMessage(content=VALIDATION_SYSTEM_PROMPT),
        HumanMessage(content=content)
    ]

    result = structured_llm.invoke(messages)

    return State(
        **{
            **state.__dict__,
            "validation": ValidationState(
                passed=result.passed,
                discrepancies=result.discrepancies,
                matches=result.matches
            ),
            "messages": state.messages + [AIMessage(content=str(result))]
        }
    )


def should_retry(state: State) -> Literal["retry", "continue", "end"]:
    """Decide if we need to regenerate code based on errors or failed validation."""
    if state.parse_error:
        return "retry" if len(state.messages) < 6 else "end"
    if state.compile_errors:
        return "retry" if len(state.messages) < 6 else "end"
    if state.validation and not state.validation.passed:
        return "retry" if len(state.messages) < 6 else "end"
    return "continue"


def generate_code(state: State, config: RunnableConfig) -> State:
    """Use the LLM to generate or fix the React component code."""
    model = ChatOpenAI(model="gpt-4o")

    # Optional example: show previously discovered doc references or relevant context
    research_context = ""
    if state.relevant_docs:
        analysis = state.relevant_docs[0].get("analysis", [])
        if analysis:
            research_context = "\nTechnical considerations from docs:\n" + "\n".join(f"- {item}" for item in analysis)

    # Summarize current code or errors
    current_code = ""
    if state.generated_code:
        current_code = "Current code:\n"
        for f in state.generated_code:
            current_code += f"\n{f["filename"]}:\n```\n{f["content"]}\n```\n"

    if state.parse_error:
        prompt = f"The code extraction failed: {state.parse_error}. Please provide valid component code.\n{research_context}\n{current_code}"
    elif state.compile_errors:
        prompt = f"The component failed to compile:\n{state.compile_errors}\nPlease fix.\n{research_context}\n{current_code}"
    elif state.validation and not state.validation.passed:
        disc = "\n".join([f"- {d}" for d in (state.validation.discrepancies or [])])
        mts = "\n".join([f"- {m}" for m in (state.validation.matches or [])])
        prompt = (
            f"The component doesn't match the design.\nDISCREPANCIES:\n{disc}\nMATCHES:\n{mts}\n"
            f"{research_context}\n{current_code}\nPlease fix it to match the requirements."
        )
    else:
        prompt = (
            f"Create a React component using Material-UI. It must match the desktop and mobile designs perfectly.\n"
            f"Requirements:\n{state.requirements}\n{research_context}"
        )

    message_content = [
        {
            "type": "text",
            "text": prompt
        }
    ]

    # Add relevant design screenshots to the context
    if state.desktop_design_screenshot:
        message_content.extend([
            {"type": "text", "text": "Desktop design screenshot:"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{detect_media_type(state.desktop_design_screenshot)};base64,{state.desktop_design_screenshot}",
                    "detail": "high"
                }
            }
        ])
    if state.component_screenshots.get("desktop"):
        message_content.extend([
            {"type": "text", "text": "Current Desktop Implementation:"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{detect_media_type(state.component_screenshots['desktop'])};base64,{state.component_screenshots['desktop']}",
                    "detail": "high"
                }
            }
        ])
    if state.mobile_design_screenshot:
        message_content.extend([
            {"type": "text", "text": "Mobile design screenshot:"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{detect_media_type(state.mobile_design_screenshot)};base64,{state.mobile_design_screenshot}",
                    "detail": "high"
                }
            }
        ])
    if state.component_screenshots.get("mobile"):
        message_content.extend([
            {"type": "text", "text": "Current Mobile Implementation:"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{detect_media_type(state.component_screenshots['mobile'])};base64,{state.component_screenshots['mobile']}",
                    "detail": "high"
                }
            }
        ])

    response = model.invoke([
        SystemMessage(content=REACT_DEVELOPER_SYSTEM_PROMPT),
        HumanMessage(content=message_content)
    ])

    return {
        "messages": state.messages + [response]
    }


async def package_component(state: State, config: RunnableConfig) -> State:
    """
    Package the component into a zip file if all steps are successful.
    """
    if (state.validation and not state.validation.passed) or state.compile_errors or state.parse_error:
        return State(
            **{
                **state.__dict__,
                "output": OutputState(
                    success=False,
                    error_message="Component generation failed or did not pass validation."
                )
            }
        )

    configuration = Configuration.from_runnable_config(config)
    temp_dir = Path(zipfile.tempfile.mkdtemp())

    try:
        component_dir = temp_dir / "component"
        component_dir.mkdir(parents=True)

        # Save screenshots if any
        screenshots_dir = component_dir / "screenshots"
        screenshots_dir.mkdir()
        screenshot_paths = {}

        if state.component_screenshots:
            for view_type, base64_data in state.component_screenshots.items():
                screenshot_path = screenshots_dir / f"{view_type}.png"
                with open(screenshot_path, 'wb') as f:
                    f.write(base64.b64decode(base64_data))
                screenshot_paths[view_type] = str(screenshot_path.relative_to(component_dir))

        # Write code files
        for file in state.generated_code:
            (component_dir / file.filename).write_text(file.content, encoding='utf-8')

        metadata = ComponentMetadata(
            name="GeneratedComponent",
            description=state.requirements,
            props={},
            screenshots=screenshot_paths,
            source_files={
                file.filename: str((component_dir / file.filename).relative_to(component_dir))
                for file in state.generated_code
            }
        )

        configuration.component_output_dir.mkdir(parents=True, exist_ok=True)
        zip_path = configuration.component_output_dir / "component.zip"

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in component_dir.rglob('*'):
                if file.is_file():
                    zipf.write(file, file.relative_to(component_dir))

        return State(
            **{
                **state.__dict__,
                "output": OutputState(
                    component_zip=zip_path,
                    metadata=metadata,
                    success=True
                )
            }
        )

    except Exception as e:
        return State(
            **{
                **state.__dict__,
                "output": OutputState(
                    success=False,
                    error_message=(
                        f"Failed to package component: {str(e)}\n\nStack trace:\n{traceback.format_exc()}"
                    )
                )
            }
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# Define the state graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Main workflow
builder.add_node("research", research)
builder.add_node("generate_code", generate_code)
builder.add_node("extract", extract)
builder.add_node("compile", compile)
builder.add_node("validate", validate)
builder.add_node("package", package_component)

builder.add_edge(START, "research")
builder.add_edge("research", "generate_code")
builder.add_edge("generate_code", "extract")
builder.add_edge("extract", "compile")
builder.add_edge("compile", "validate")
builder.add_edge("validate", "package")

# Retry logic
builder.add_conditional_edges(
    "validate",
    should_retry,
    {
        "retry": "generate_code",
        "continue": "package",
        "end": END
    }
)

builder.add_edge("package", END)

graph = builder.compile()
graph.name = "React Component Developer"
