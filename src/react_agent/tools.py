"""Tools for React component generation and testing."""

import subprocess
import sys
import os
from typing import Dict, Any, List, Optional
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, BaseTool, tool
from typing_extensions import Annotated

from react_agent.configuration import Configuration
from react_agent.dev_server import DevServer
from react_agent.models import FileItem, SearchResult, FileContents

# Global variable to track current directory during controller file search
_current_search_directory = None
_repo_root_path = None

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


def setup_repo_search(repo_path: str) -> None:
    """Set up repository search by initializing the current directory."""
    global _current_search_directory, _repo_root_path
    _repo_root_path = repo_path
    _current_search_directory = repo_path


@tool
def navigate_directory(path: str) -> str:
    """
    Change to a specified directory within the repository.
    
    Args:
        path: Path to navigate to, can be relative to current directory or absolute from repo root
        
    Returns:
        String describing the result of the navigation and contents of the new directory.
    """
    global _current_search_directory
    
    if not _current_search_directory:
        return "Error: Repository search has not been initialized."
    
    # Handle absolute vs relative paths
    if path.startswith('/'):
        new_dir = os.path.join(_repo_root_path, path.lstrip('/'))
    else:
        new_dir = os.path.normpath(os.path.join(_current_search_directory, path))
    
    # Verify the directory exists
    if not os.path.exists(new_dir):
        return f"Error: Directory '{path}' does not exist"
    if not os.path.isdir(new_dir):
        return f"Error: Path '{path}' is not a directory"
    
    # Update current directory
    _current_search_directory = new_dir
    relative_path = os.path.relpath(new_dir, _repo_root_path)
    if relative_path == '.':
        relative_path = '/'
    else:
        relative_path = '/' + relative_path
    
    # Get directory contents
    dirs, files = [], []
    for entry in sorted(os.listdir(new_dir)):
        entry_path = os.path.join(new_dir, entry)
        # Skip hidden files/dirs
        if entry.startswith('.'):
            continue
        # Skip large dirs like node_modules
        if os.path.isdir(entry_path) and entry in ['node_modules', 'dist', 'build', 'coverage']:
            continue
        
        if os.path.isdir(entry_path):
            dirs.append(f"{entry}/")
        else:
            files.append(entry)
    
    # Format the result
    result = f"Changed directory to {relative_path}\n\nDirectories:\n"
    result += "\n".join(dirs) if dirs else "No directories"
    result += "\n\nFiles:\n"
    result += "\n".join(files) if files else "No files"
    
    return result


@tool
def list_directory(path: Optional[str] = None) -> str:
    """
    List contents of a directory.
    
    Args:
        path: Optional path to list. If not provided, lists the current directory.
        
    Returns:
        String representation of directory contents.
    """
    global _current_search_directory
    
    if not _current_search_directory:
        return "Error: Repository search has not been initialized."
    
    # Determine which directory to list
    if path:
        if path.startswith('/'):
            dir_to_list = os.path.join(_repo_root_path, path.lstrip('/'))
        else:
            dir_to_list = os.path.normpath(os.path.join(_current_search_directory, path))
    else:
        dir_to_list = _current_search_directory
        path = os.path.relpath(dir_to_list, _repo_root_path)
        if path == '.':
            path = '/'
        else:
            path = '/' + path
    
    # Verify the directory exists
    if not os.path.exists(dir_to_list):
        return f"Error: Directory '{path}' does not exist"
    if not os.path.isdir(dir_to_list):
        return f"Error: Path '{path}' is not a directory"
    
    # Get directory contents
    dirs, files = [], []
    for entry in sorted(os.listdir(dir_to_list)):
        entry_path = os.path.join(dir_to_list, entry)
        # Skip hidden files/dirs
        if entry.startswith('.'):
            continue
        # Skip large dirs like node_modules
        if os.path.isdir(entry_path) and entry in ['node_modules', 'dist', 'build', 'coverage']:
            continue
        
        if os.path.isdir(entry_path):
            dirs.append(f"{entry}/")
        else:
            files.append(entry)
    
    # Format the result
    result = f"Contents of {path}:\n\nDirectories:\n"
    result += "\n".join(dirs) if dirs else "No directories"
    result += "\n\nFiles:\n"
    result += "\n".join(files) if files else "No files"
    
    return result


@tool
def read_file(path: str) -> str:
    """
    Read the content of a file.
    
    Args:
        path: Path to the file to read, relative to current directory or absolute from repo root
        
    Returns:
        Content of the file or error message.
    """
    global _current_search_directory
    
    if not _current_search_directory:
        return "Error: Repository search has not been initialized."
    
    # Normalize path handling
    if path.startswith('/'):
        full_path = os.path.join(_repo_root_path, path.lstrip('/'))
    else:
        full_path = os.path.normpath(os.path.join(_current_search_directory, path))
    
    # Verify the file exists
    if not os.path.exists(full_path):
        return f"Error: File '{path}' does not exist"
    if not os.path.isfile(full_path):
        return f"Error: Path '{path}' is not a file"
    
    try:
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        if len(content) > 10000:
            content = content[:10000] + "\n... (content truncated, file too large)"
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def find_controller(path: str, confidence: int, reasoning: str) -> Dict[str, Any]:
    """
    Declare that a controller file has been found.
    
    Args:
        path: Path to the controller file, relative to repo root
        confidence: Confidence level (1-10) that this is the right controller
        reasoning: Explanation of why this is believed to be the controller file
        
    Returns:
        Dictionary with result of the declaration.
    """
    global _repo_root_path
    
    if not _repo_root_path:
        return {"success": False, "error": "Repository search has not been initialized."}
    
    # Standardize path handling
    if not path.startswith('/'):
        # Convert to absolute path from repo root
        path = '/' + path
    
    # Verify the file exists
    full_path = os.path.join(_repo_root_path, path.lstrip('/'))
    if not os.path.exists(full_path):
        return {
            "success": False, 
            "error": f"File '{path}' does not exist"
        }
    
    if not os.path.isfile(full_path):
        return {
            "success": False, 
            "error": f"Path '{path}' is not a file"
        }
    
    return {
        "success": True,
        "file_path": str(Path(full_path)),
        "confidence": confidence,
        "reasoning": reasoning
    }


# Include file navigation tools in the TOOLS list
TOOLS = [
    search_docs,
    compile_component,
    take_screenshots,
    navigate_directory,
    list_directory,
    read_file,
    find_controller
]
