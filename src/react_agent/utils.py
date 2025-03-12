"""Utility & helper functions."""

import base64
import subprocess
import sys
import os
from typing import Dict, Any, List

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from react_agent.ChaseAzureOpenAI import getModel

from react_agent.prompts import EXTRACTION_SYSTEM_PROMPT

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated

from react_agent.configuration import Configuration
from react_agent.dev_server import DevServer
from react_agent.models import FileItem, FileContents

from playwright.sync_api import sync_playwright

def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()

def extract_code(content: str) -> FileContents:
    """Extract JavaScript/TypeScript and CSS code from markdown content using an LLM."""
    model = getModel()
    structured_llm = model.with_structured_output(FileContents)

    messages = [
        SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=f"Please extract and organize the code from the following text:\n\n{content}"),
    ]
    
    return structured_llm.invoke(messages)

def detect_media_type(base64_data: str) -> str:
    """Detect the media type from base64 data."""
    # Check the first few bytes of the decoded data
    try:
        import magic
        decoded = base64.b64decode(base64_data)
        return magic.from_buffer(decoded, mime=True)
    except ImportError:
        # Fallback: Check common image signatures
        try:
            decoded = base64.b64decode(base64_data[:32])  # First few bytes are enough
            if decoded.startswith(b'\x89PNG\r\n\x1a\n'):
                return 'image/png'
            elif decoded.startswith(b'\xff\xd8'):
                return 'image/jpeg'
            elif decoded.startswith(b'GIF87a') or decoded.startswith(b'GIF89a'):
                return 'image/gif'
            elif decoded.startswith(b'RIFF') and decoded[8:12] == b'WEBP':
                return 'image/webp'
            else:
                return 'image/png'  # Default fallback
        except:
            return 'image/png'  # Default fallback

def compile_component(
    component_code: List[FileItem],
    dependencies: List[str],
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> Dict[str, Any]:
    """
    Compile the extracted React component using a temporary dev server.
    Now it expects a top-level project_metadata.dependencies array for npm packages.
    """
    server = None
    try:
        # Convert the extracted items into the format DevServer expects
        files_for_dev_server = []
        for item in component_code:
            # If it's the entrypoint, rename to "component/index.tsx"
            # or use item.filename verbatim in "component/" subfolder:
            filename = (
                "component/index.tsx"
                if item["entrypoint"] else
                f"component/{item["filename"]}"
            )
            
            # Create FileItem instance the correct way
            file_item = FileItem()
            file_item.filename = filename
            file_item.content = item["content"]
            file_item.file_type = item["file_type"]
            file_item.entrypoint = item["entrypoint"]
            
            files_for_dev_server.append(file_item)

        server = DevServer(port=Configuration.from_runnable_config(config).dev_server_port)
        server.setup(
            {"files": files_for_dev_server},
            dependencies=dependencies
        )

        build_result = server.build()

        if not build_result["success"]:
            return {
                "success": False,
                "errors": [build_result["message"]],
                "server": server
            }

        result = server.start()
        if not result["success"]:
            return {
                "success": False,
                "errors": result["errors"],
                "server": server
            }

        return {
            "success": True,
            "dev_server_url": result["dev_server_url"],
            "server": server
        }

    except Exception as e:
        return {
            "success": False, 
            "errors": [str(e)]
        }

def get_console_errors(url):
    # Create a new instance of the Playwright driver
    with sync_playwright() as playwright:
        # Launch a new browser
        browser = playwright.chromium.launch()

        # Create a new page
        page = browser.new_page()

        # Navigate to the URL
        page.goto(url)

        # Get the console errors
        console_errors = page.evaluate('() => { return console.error.toString() }')

        # Close the browser
        browser.close()

        # Return the console errors if there are any, else return an empty string
        if console_errors:
            return "ClientSide Console Errors Occurred:\n" + console_errors
        else:
            return ""

def _take_screenshots_process(url: str, timeout: int = 6) -> Dict[str, str]:
    """
    Run Playwright in a subprocess to take desktop and mobile screenshots.
    """
    script = f"""
import base64
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

output = {{}}

try:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={{'width': 1280, 'height': 800}})
        page.goto("{url}")
        page.wait_for_timeout(1000)
        desktop_bytes = page.screenshot(type='png')
        output['desktop'] = base64.b64encode(desktop_bytes).decode('utf-8')

        page.set_viewport_size({{'width': 375, 'height': 667}})
        page.wait_for_timeout(500)
        mobile_bytes = page.screenshot(type='png')
        output['mobile'] = base64.b64encode(mobile_bytes).decode('utf-8')

        browser.close()
except Exception as e:
    output['errors'] = str(e)

print(repr(output))
"""

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout + 1
        )
        if result.returncode == 0:
            return eval(result.stdout.strip())
        else:
            return {"errors": f"Screenshot process failed: {result.stderr}"}
    except subprocess.TimeoutExpired:
        return {"errors": f"Screenshot capture timed out after {timeout} seconds"}
    except Exception as e:
        return {"errors": f"Screenshot capture failed: {str(e)}"}
    finally:
        try:
            os.unlink(temp_path)
        except:
            pass


def take_screenshots(
    url: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> Dict[str, str]:
    """Take screenshots of the running component in desktop and mobile viewports."""
    return _take_screenshots_process(url, timeout=6)


if __name__ == "__main__":
    print(get_console_errors("https://www.chase.com"))