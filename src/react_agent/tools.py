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
    Change the current directory.
    
    Args:
        path: Path to navigate to. Can be absolute (starting with '/') or relative to current directory.
              Supports '..' to go up one directory level and '.' to reference current directory.
        
    Returns:
        String confirming the navigation or an error message.
    """
    global _current_search_directory, _repo_root_path
    
    if not _current_search_directory:
        return "Error: Repository search has not been initialized."
    
    try:
        # Handle absolute paths (starting with /)
        if path.startswith('/'):
            new_abs_path = os.path.normpath(os.path.join(_repo_root_path, path.lstrip('/')))
        else:
            # Handle relative paths (including . and .. navigation)
            new_abs_path = os.path.normpath(os.path.join(_current_search_directory, path))
        
        # Ensure we can't navigate outside the repo root
        if not new_abs_path.startswith(_repo_root_path):
            return f"Error: Cannot navigate outside the repository root."
        
        # Verify the directory exists
        if not os.path.exists(new_abs_path):
            return f"Error: Directory '{path}' does not exist"
        if not os.path.isdir(new_abs_path):
            return f"Error: Path '{path}' is not a directory"
        
        # Update the current directory
        _current_search_directory = new_abs_path
        
        # Calculate relative path from repo root for display
        rel_path = os.path.relpath(_current_search_directory, _repo_root_path)
        display_path = '/' + rel_path if rel_path != '.' else '/'
        
        # Get directory contents for the new location
        contents = list_directory('.')
        
        return f"Changed directory to {display_path}\n\nContents:\n{contents}"
    
    except Exception as e:
        return f"Error navigating to directory: {str(e)}"


@tool
def list_directory(path: str = '.') -> str:
    """
    List contents of a directory.
    
    Args:
        path: Path to list, relative to current directory or absolute from repo root.
              Defaults to current directory ('.').
              Supports '..' to reference parent directory and '.' for current directory.
        
    Returns:
        String listing of directory contents or an error message.
    """
    global _current_search_directory, _repo_root_path
    
    if not _current_search_directory:
        return "Error: Repository search has not been initialized."
    
    try:
        # Handle absolute paths
        if path.startswith('/'):
            full_path = os.path.normpath(os.path.join(_repo_root_path, path.lstrip('/')))
        else:
            # Handle relative paths (including . and ..)
            full_path = os.path.normpath(os.path.join(_current_search_directory, path))
        
        # Ensure we don't list outside the repo root
        if not full_path.startswith(_repo_root_path):
            return f"Error: Cannot list directories outside the repository root."
        
        # Verify the directory exists
        if not os.path.exists(full_path):
            return f"Error: Path '{path}' does not exist"
        if not os.path.isdir(full_path):
            return f"Error: Path '{path}' is not a directory"
        
        # Calculate relative path from repo root for display
        rel_path = os.path.relpath(full_path, _repo_root_path)
        display_path = '/' + rel_path if rel_path != '.' else '/'
        
        # Get directory contents
        entries = sorted(os.listdir(full_path))
        
        # Organize into dirs and files
        dirs = []
        files = []
        for entry in entries:
            entry_path = os.path.join(full_path, entry)
            if os.path.isdir(entry_path):
                dirs.append(f"{entry}/")
            else:
                files.append(entry)
        
        # Skip certain directories if they exist
        skip_dirs = ['node_modules', '.git', 'dist', 'build']
        dirs = [d for d in dirs if d.rstrip('/') not in skip_dirs]
        
        # Format the output
        output = [f"Contents of {display_path}:"]
        
        if dirs:
            output.append("\nDirectories:")
            for d in dirs:
                output.append(f"  {d}")
        
        if files:
            output.append("\nFiles:")
            for f in files:
                output.append(f"  {f}")
        
        if not dirs and not files:
            output.append("\nEmpty directory")
        
        return "\n".join(output)
    
    except Exception as e:
        return f"Error listing directory: {str(e)}"


@tool
def read_file(path: str) -> str:
    """
    Read the content of a file.
    
    Args:
        path: Path to the file to read, relative to current directory or absolute from repo root.
              Supports '..' to go up one directory level and '.' to reference current directory.
              Remember that the path is relative to the current directory, not the repository root.
        
    Returns:
        Content of the file or error message.
    """
    global _current_search_directory, _repo_root_path
    
    if not _current_search_directory:
        return "Error: Repository search has not been initialized."
    
    try:
        # Normalize path handling
        if path.startswith('/'):
            full_path = os.path.normpath(os.path.join(_repo_root_path, path.lstrip('/')))
        else:
            full_path = os.path.normpath(os.path.join(_current_search_directory, path))
        
        # Ensure we can't read files outside the repo root
        if not full_path.startswith(_repo_root_path):
            return f"Error: Cannot read files outside the repository root."
        
        # Verify the file exists
        if not os.path.exists(full_path):
            return f"Error: File '{path}' does not exist"
        if not os.path.isfile(full_path):
            return f"Error: Path '{path}' is not a file"
        
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
        path: Path to the controller file, relative to current directory or absolute from repo root.
              Supports '..' to go up one directory level and '.' to reference current directory.
        confidence: Confidence level (1-10) that this is the right controller
        reasoning: Explanation of why this is believed to be the controller file
        
    Returns:
        Dictionary with result of the declaration.
    """
    global _current_search_directory, _repo_root_path
    
    if not _current_search_directory or not _repo_root_path:
        return {"success": False, "error": "Repository search has not been initialized."}
    
    try:
        # Handle absolute vs relative paths
        if path.startswith('/'):
            full_path = os.path.normpath(os.path.join(_repo_root_path, path.lstrip('/')))
        else:
            # Handle relative paths (including . and .. navigation)
            full_path = os.path.normpath(os.path.join(_current_search_directory, path))
        
        # Ensure we can't access files outside the repo root
        if not full_path.startswith(_repo_root_path):
            return {
                "success": False, 
                "error": f"Cannot access files outside the repository root."
            }
        
        # Verify the file exists
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
        
        # Calculate relative path from repo root for display
        rel_path = os.path.relpath(full_path, _repo_root_path)
        display_path = '/' + rel_path if rel_path != '.' else '/file'
        
        return {
            "success": True,
            "file_path": str(Path(full_path)),
            "display_path": display_path,
            "confidence": confidence,
            "reasoning": reasoning
        }
        
    except Exception as e:
        return {
            "success": False, 
            "error": f"Error processing controller file path: {str(e)}"
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
