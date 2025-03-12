"""Tools for React component generation and testing."""

import os
from pathlib import Path
from typing import Dict, Any, List

from react_agent.vector_store import vector_store
from react_agent.models import ResearchQuestion
from react_agent.ChaseAzureOpenAI import getModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from playwright.sync_api import sync_playwright

# Global variable to track current directory during controller file search
_current_search_directory = None
_repo_root_path = None


def search_docs(
    queries: List[ResearchQuestion],
) -> str:
    """
    This function returns the search results for the given queries against the vector store that houses our internal docs about Octagon, BlueJS, and MDS.
    It takes a list of query objects (you can ask multiple questions at once) and returns a string of the results.
    """

    model = getModel()

    result = ""

    # get answers from RAG system
    for query in queries:
        docs = vector_store.similarity_search(query.question, k=30, filter={"package_name": query.filter})
        q = f"Question: {query.question}\nChunks:\n\n"
        for doc in docs:
            q += f"{doc.page_content}\n\n"

        result += q
    
    response = model.invoke([
        SystemMessage(content="You will be provided a set of questions each with a set of chunks retreived from a vector store search. Answer each question using the context provided. If the answer isn't in the context chunks, just say there wasn't enough info found in the docs. Format in an ordered list in markdown. Be detailed and include all code examples where possible so the larger system you pass this too can know how to write the application code that will be using these answers."),
        HumanMessage(content=result),
    ])

    return response.content

def search_chase_interweb(url: str) -> str:
    """Searches the interweb for the given URL and return the page content.
    Important links:
    Octagon Libraries and API References: https://octagon-docs.dev.sch.jpmchase.net/reference/
    Octagon Getting Started: https://go/octagon/
    MDS Components Library Docs: https://manhattan.gaiacloud.jpmchase.net/get-started/for-developers#web
    MDS Components Library Overview (has list of components): https://manhattan.gaiacloud.jpmchase.net/doc/components/overview

    If pages contain links, you can use this function again to get the content of the linked pages.
    """
    
    model = getModel()

    result = ""

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        
        result = page.content()

        print(result)
        browser.close()

    response = model.invoke([
        SystemMessage(content="You will be provided html page content. Convert it to markdown format and return it. Include any links on the page as well as navigation links or tables of contents."),
        HumanMessage(content=result),
    ])

    return response.content

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

RESEARCH_TOOLS = [
    search_docs,
    search_chase_interweb
]

if __name__ == "__main__":
    print(search_chase_interweb("https://go/octagon/"))
