"""Tools for React component generation and testing."""

import subprocess
import sys
import os
from typing import Dict, Any, List

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated

from react_agent.configuration import Configuration
from react_agent.dev_server import DevServer
from react_agent.models import FileItem, SearchResult, FileContents


def search_docs(
    query: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> List[SearchResult]:
    """
    Example stub for documentation search.
    """
    return [
        SearchResult(
            content="Example MUI or relevant documentation results",
            source="docs/example.md",
            score=0.95
        )
    ]


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

        result = server.start()
        if not result["success"]:
            return {"errors": result["errors"]}

        return {
            "success": True,
            "dev_server_url": result["dev_server_url"]
        }

    except Exception as e:
        return {"errors": [str(e)]}
    
    finally:
        # Clean up the dev server and temporary files
        if server:
            server.stop()


def _take_screenshots_process(url: str) -> Dict[str, str]:
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
            timeout=30
        )
        if result.returncode == 0:
            return eval(result.stdout.strip())
        else:
            return {"errors": f"Screenshot process failed: {result.stderr}"}
    except subprocess.TimeoutExpired:
        return {"errors": "Screenshot capture timed out"}
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
    return _take_screenshots_process(url)


TOOLS = [
    search_docs,
    compile_component,
    take_screenshots
]
