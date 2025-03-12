"""Define a custom React Component Development agent."""

import base64
import json
import os
import shutil
import traceback
import zipfile
from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, START

from react_agent.configuration import Configuration
from react_agent.state import ComponentMetadata, OutputState, State, InputState, ValidationState
from react_agent.tools import search_docs, RESEARCH_TOOLS
from react_agent.utils import extract_code, detect_media_type, compile_component, take_screenshots, get_console_errors
from react_agent.prompts import (
    VALIDATION_SYSTEM_PROMPT,
    REACT_DEVELOPER_SYSTEM_PROMPT,
    RESEARCH_SYSTEM_PROMPT
)
from react_agent.models import ValidationResult
from react_agent.ChaseAzureOpenAI import getModel
from langgraph.prebuilt import ToolNode


def research(state: State, config: RunnableConfig) -> State:
    """Research needed docs from vector db by analyzing designs and requirements."""
    model = getModel()
    structured_llm = model.bind_tools(RESEARCH_TOOLS)

    if len(state.research_agent_messages) > 0:
        last_message = state.research_agent_messages[-1]
    else:
        last_message = None
    
    # Create message content for GPT-4V analysis
    content = [
        {
            "type": "text",
            "text": f"Requirements:\n{state.requirements}"
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
    
    metadataValues = {"package_name": []}

    # load metadata file if it exists
    if os.path.exists("metadata.json"):
        with open('metadata.json') as json_file:
            metadataValues = json.load(json_file)

    messages = [
        SystemMessage(content=RESEARCH_SYSTEM_PROMPT + f"\nAvailable metadata filters: {metadataValues['package_name']}"),
        *state.research_agent_messages
    ]

    # add HumanMessage only if last message is not ToolMessage
    if last_message == None or not isinstance(last_message, ToolMessage):
        messages.append(HumanMessage(content=content))
    
    # Get structured analysis and questions from GPT-4V
    result = structured_llm.invoke(messages)
    
    # Store the analysis and questions
    return State(
        **{
            **state.__dict__,
            "research_agent_messages": [*state.research_agent_messages, HumanMessage(content=content), result],
            "relevant_docs": result.content
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
            print("Taking screenshots at " + result["dev_server_url"])
            console_errors = get_console_errors(result["dev_server_url"])

            if console_errors:
                return State(
                    **{
                        **state.__dict__,
                        "compile_errors": [console_errors]
                    }
                )

            screenshots = take_screenshots(result["dev_server_url"], config=config)
            result["server"].stop()
            print("Stopped server")

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
    if state.compile_errors:
        return State(
            **{
                **state.__dict__,
                "validation": {
                    "passed": False,
                    "discrepancies": ["Compile Step Failed with the following errors:\n" + "\n".join(state.compile_errors)],
                    "matches": []
                }
            }
        )
    
    if not state.component_screenshots or not state.desktop_design_screenshot:
        return State(
            **{
                **state.__dict__,
                "validation": {
                    "passed": False, "discrepancies": ["Missing screenshots for validation"]},
                    "discrepancies": ["Missing screenshots for validation"],
                    "matches": []
            }
        )

    model = getModel()
    structured_llm = model.with_structured_output(ValidationResult)

    # Prepare comparison images
    content = [
        {
            "type": "text",
            "text": f"Requirements:\n{state.requirements}"
        },
        {
            "type": "text",
            "text": "Generated Code:\n" + "\n".join([f"{f['filename']}: {f['content']}" for f in state.generated_code])
        }
    ]

    if state.desktop_design_screenshot:
        content.extend([
            {
                "type": "text",
                "text": "\n\nOriginal Desktop Design:"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{detect_media_type(state.desktop_design_screenshot)};base64,{state.desktop_design_screenshot}",
                    "detail": "high"
                }
            }
        ])

    if state.mobile_design_screenshot:
        content.extend([
            {
                "type": "text",
                "text": "\n\nOriginal Mobile Design:"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{detect_media_type(state.mobile_design_screenshot)};base64,{state.mobile_design_screenshot}",
                    "detail": "high"
                }
            }
        ])

    if state.component_screenshots:
        if "desktop" in state.component_screenshots:
            content.extend([
                {
                    "type": "text",
                    "text": "\n\nImplemented Component (Desktop):"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{detect_media_type(state.component_screenshots['desktop'])};base64,{state.component_screenshots['desktop']}",
                        "detail": "high"
                    }
                }
            ])

        if "mobile" in state.component_screenshots:
            content.extend([
                {
                    "type": "text",
                    "text": "\n\nImplemented Component (Mobile):"
                },
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
            "validation": {
                "passed": result.passed,
                "discrepancies": result.discrepancies,
                "matches": result.matches
            }
        }
    )


def generate_code(state: State, config: RunnableConfig) -> State:
    """Use the LLM to generate or fix the React component code."""
    model = getModel().bind_tools(RESEARCH_TOOLS)

    if len(state.research_agent_messages) > 0:
        last_message = state.research_agent_messages[-1]
    else:
        last_message = None

    # Optional example: show previously discovered doc references or relevant context
    research_context = ""
    if state.relevant_docs:
        research_context = f"\nTechnical considerations from docs:\n{state.relevant_docs}"

    # Summarize current code or errors
    current_code = ""
    if state.generated_code:
        current_code = "Current code:\n"
        for f in state.generated_code:
            current_code += f"\n{f["filename"]}:\n```\n{f["content"]}\n```\n"

    if state.parse_error:
        prompt = f"The code extraction failed: {state.parse_error}. Please provide valid component code.\n{research_context}\n{current_code}"
    elif state.validation and not state.validation['passed']:
        disc = "\n".join([f"- {d}" for d in (state.validation['discrepancies'] or [])])
        mts = "\n".join([f"- {m}" for m in (state.validation['matches'] or [])])
        prompt = (
            "Validation Results:\n"
            f"\nDISCREPANCIES:\n{disc}\nMATCHES:\n{mts}\n"
            f"{research_context}\n{current_code}\nPlease fix it to match the requirements."
        )
    else:
        prompt = (
            f"Create a React component using J.P. Morgan Chase's Octagon React Framework/libraries and their MDS Component Library. It must match the desktop and mobile designs perfectly.\n"
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
    
    # add HumanMessage only if last message is not ToolMessage
    if last_message == None or not isinstance(last_message, ToolMessage):
        response = model.invoke([
            SystemMessage(content=REACT_DEVELOPER_SYSTEM_PROMPT),
            *state.messages,
            HumanMessage(content=message_content)
        ])
    else:
        response = model.invoke([
            SystemMessage(content=REACT_DEVELOPER_SYSTEM_PROMPT),
            *state.messages
        ])

    return {
        "messages": state.messages + [response],
        "validation": {
            "passed": False,
            "discrepancies": [],
            "matches": []
        },
        "compile_errors": [],
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


def is_research_toolcall(state: State) -> Literal["next", "return_to"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("next" or "return_to").
    """
    last_message = state.research_agent_messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "next"
    # Otherwise we execute the requested actions
    return "is_tool_call"

def is_generation_toolcall(state: State) -> Literal["next", "return_to"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("next" or "return_to").
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "next"
    # Otherwise we execute the requested actions
    return "is_tool_call"

def should_retry(state: State) -> Literal["retry", "continue", "end"]:
    """Decide if we need to regenerate code based on errors or failed validation."""
    if state.parse_error:
        return "retry" if len(state.messages) < 6 else "end"
    if state.compile_errors:
        return "retry" if len(state.messages) < 6 else "end"
    if state.validation and not state.validation.passed:
        return "retry" if len(state.messages) < 6 else "end"
    return "continue"


# Define the state graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Main workflow
builder.add_node("research", research)
builder.add_node("research_tools", ToolNode(tools=RESEARCH_TOOLS, messages_key="research_agent_messages"))
builder.add_node("generate_code", generate_code)
builder.add_node("generation_tools", ToolNode(tools=RESEARCH_TOOLS))
builder.add_node("extract", extract)
builder.add_node("compile", compile)
builder.add_node("validate", validate)
builder.add_node("package", package_component)

builder.add_edge(START, "research")
builder.add_edge("research_tools", "research")

builder.add_edge("generation_tools", "generate_code")

builder.add_edge("extract", "compile")
builder.add_edge("compile", "validate")

builder.add_conditional_edges(
    "generate_code",
    is_generation_toolcall,
    {
        "next": "extract",
        "is_tool_call": "generation_tools",
    }
)

builder.add_conditional_edges(
    "research",
    # After research finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    is_research_toolcall,
    {
        "next": "generate_code",
        "is_tool_call": "research_tools",
    }
)

# Retry logic
builder.add_conditional_edges(
    "validate",
    should_retry,
    {
        "retry": "generate_code",
        "continue": "package",
        "end": END,
    }
)

builder.add_edge("package", END)

graph = builder.compile()
graph.name = "React Component Developer"
