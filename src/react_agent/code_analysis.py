"""
Code analysis module to extract requirements and convert BlueJS applications to React.
"""

import os
import subprocess
import shutil
from typing import Dict, List, Tuple, Optional, Any, Set
import re
from pathlib import Path
import json
import time
import logging
import hashlib

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from react_agent.ChaseAzureOpenAI import getModel
from langchain.memory import ConversationBufferMemory
from react_agent.tools import (
    setup_repo_search, navigate_directory, list_directory, 
    read_file, find_controller
)
from react_agent.models import DependencyList

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define a dedicated location for storing repositories
REPO_STORAGE_DIR = os.path.join(os.path.expanduser("~"), ".react_agent", "repositories")

class CodeAnalyzer:
    """Analyzes BlueJS code repositories to extract requirements and convert to React."""
    
    def __init__(self, model_name: str = "gpt-4o", timeout_minutes: int = 30):
        """Initialize with the specified LLM model."""
        self.llm = getModel()
        # Add memory to track the conversation with the LLM
        self.memory = ConversationBufferMemory(return_messages=True)
        # Track visited files to avoid circular dependencies
        self.visited_files = set()
        # Map of module aliases to actual file paths
        self.alias_map = {}
        # Keep track of dependencies between files
        self.dependency_graph = {}
        # Set timeout
        self.timeout = timeout_minutes * 60
        self.start_time = None
        # For caching large responses
        self.analysis_cache = {}
    
    def analyze_repo(self, repo_url: str, screen_id: str) -> Dict[str, Any]:
        """
        Main entry point to analyze a repository based on a screen ID.
        
        Args:
            repo_url: URL of the git repository to clone and analyze
            screen_id: Path-like string in format "<app_name>/<area_name>/<controller_file_name>/<action>"
            
        Returns:
            Dictionary containing analysis results, requirements and converted React components
        """
        self.start_time = time.time()
        logger.info(f"Starting analysis of {repo_url} for screen {screen_id}")
        
        # Parse the screen ID
        app_name, area_name, controller_name, action = self._parse_screen_id(screen_id)
        if action == "":
            action = "index"  # Default action
            
        logger.info(f"Parsed screen ID: App={app_name}, Area={area_name}, Controller={controller_name}, Action={action}")
        
        # Ensure the repository storage directory exists
        os.makedirs(REPO_STORAGE_DIR, exist_ok=True)
        
        # Clone the repository to our dedicated storage location
        repo_path = self._clone_repo(repo_url, REPO_STORAGE_DIR)
        
        # Generate file tree
        file_tree = self._build_file_tree(repo_path)
        logger.info("Generated file tree structure")
        
        # Find and parse configuration files for path aliases
        self._parse_config_files(repo_path)
        logger.info(f"Parsed configuration files for path aliases. Found {len(self.alias_map)} aliases.")
        
        # Find controller file using LLM
        controller_file = self._find_controller_file_with_llm(repo_path, screen_id)
        if not controller_file:
            # Fall back to heuristic-based search
            logger.info("LLM-based controller file search failed, falling back to heuristic search")
            controller_file = self._find_controller_file_with_heuristic(repo_path, screen_id)
        
        if not controller_file:
            raise ValueError(f"Could not find controller file for {controller_name}")
        
        logger.info(f"Found controller file: {controller_file}")
        
        # Analyze the controller file and its dependencies
        controller_analysis = self._analyze_file_dependencies(controller_file, repo_path)
        
        # Analyze code to generate requirements with dependency context
        requirements = self._generate_requirements_with_context(
            repo_path, 
            controller_file,
            file_tree,
            controller_analysis
        )
        
        elapsed_time = time.time() - self.start_time
        logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
        
        return {
            "screen_id": screen_id,
            "file_tree": file_tree,
            "controller_file": str(controller_file),
            "action_function": action,
            "requirements": requirements,
            "dependencies": self.dependency_graph,
            "path_aliases": self.alias_map,
            "analysis_time_seconds": elapsed_time
        }
            
    def _parse_screen_id(self, screen_id: str) -> Tuple[str, str, str, str]:
        """Parse the screen ID into its components."""
        parts = screen_id.split('/')
        if len(parts) < 3:
            raise ValueError(f"Invalid screen ID format: {screen_id}")
            
        app_name = parts[0]
        area_name = parts[1]
        controller_name = parts[2]
        action = parts[3] if len(parts) > 3 else ""
        
        return app_name, area_name, controller_name, action
    
    def _clone_repo(self, repo_url: str, target_dir: str) -> str:
        """
        Clone the git repository to the target directory if it doesn't already exist.
        If the repo was already cloned previously, reuse the existing directory.
        """
        # Create a consistent directory name based on the repo URL
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        repo_hash = hashlib.md5(repo_url.encode()).hexdigest()[:8]
        repo_dir = os.path.join(target_dir, f"{repo_name}_{repo_hash}")
        
        # Check if the directory already exists and is a valid git repo
        if os.path.exists(repo_dir):
            logger.info(f"Repository directory already exists: {repo_dir}")
            
            # Check if it's a valid git repository
            try:
                # Try running git status to see if it's a valid repo
                result = subprocess.run(
                    ["git", "-C", repo_dir, "status"],
                    check=True,
                    capture_output=True,
                    text=True
                )
                
                # If we get here without an exception, it's a valid git repo
                logger.info("Existing repository is valid. Pulling latest changes...")
                
                # Pull the latest changes
                subprocess.run(
                    ["git", "-C", repo_dir, "pull"],
                    check=True,
                    capture_output=True,
                    text=True
                )
                
                logger.info("Using existing repository with latest changes")
                return repo_dir
                
            except subprocess.CalledProcessError:
                logger.warning(f"Existing directory is not a valid git repository. Will clone again.")
                # Clean up the invalid directory
                shutil.rmtree(repo_dir, ignore_errors=True)
        
        # Clone the repository if we don't have a valid existing one
        logger.info(f"Cloning repository: {repo_url} to {repo_dir}")
        subprocess.run(
            ["git", "clone", repo_url, repo_dir],
            check=True,
            capture_output=True,
            text=True
        )
        return repo_dir
    
    def _build_file_tree(self, repo_path: str, max_depth: int = 6) -> str:
        """
        Build a recursive file tree of the repository to help the LLM understand the structure.
        Limits depth and excludes certain directories to avoid creating an excessively large tree.
        
        Args:
            repo_path: Path to the repository root
            max_depth: Maximum depth for the tree
            
        Returns:
            String representation of the file tree
        """
        logger.info(f"Building file tree for {repo_path}")
        
        # Directories to exclude
        excluded_dirs = {
            'node_modules', '.git', 'dist', 'build', 'coverage', 
            'test', 'tests', '__tests__', 'fixtures', '__fixtures__',
            '__mocks__', '__snapshots__'
        }
        
        # File extensions to include (focus on JavaScript/TypeScript files and configs)
        included_extensions = {
            '.js', '.jsx', '.ts', '.tsx', '.json', '.html', '.css', '.scss', '.less',
            '.config.js', '.config.ts'
        }
        
        result = []
        
        def _build_tree(dir_path, prefix="", depth=0):
            if depth > max_depth:
                result.append(f"{prefix}... (max depth reached)")
                return
            
            try:
                entries = list(os.scandir(dir_path))
                entries.sort(key=lambda e: (not e.is_dir(), e.name.lower()))
                
                # Count dirs and files for summarization
                dirs = [e for e in entries if e.is_dir() and not e.name.startswith('.') and e.name.lower() not in excluded_dirs]
                files = [e for e in entries if e.is_file() and any(e.name.endswith(ext) for ext in included_extensions)]
                
                # If there are too many entries, summarize
                if len(dirs) + len(files) > 50 and depth > 2:
                    result.append(f"{prefix}{os.path.basename(dir_path)}/ ({len(dirs)} directories, {len(files)} files)")
                    
                    # List the first 10 directories and 10 files as examples
                    for i, entry in enumerate(dirs[:10]):
                        if i == 9 and len(dirs) > 10:
                            result.append(f"{prefix}    ... ({len(dirs) - 9} more directories)")
                        else:
                            _build_tree(entry.path, prefix + "    ", depth + 1)
                    
                    for i, entry in enumerate(files[:10]):
                        if i == 9 and len(files) > 10:
                            result.append(f"{prefix}    ... ({len(files) - 9} more files)")
                        else:
                            result.append(f"{prefix}    {entry.name}")
                    
                    return
                
                # Process each entry
                for i, entry in enumerate(entries):
                    # Check if this is the last entry in the current directory
                    is_last = (i == len(entries) - 1)
                    
                    if entry.is_dir():
                        # Skip excluded directories
                        dir_name = entry.name.lower()
                        if dir_name.startswith('.') or dir_name in excluded_dirs:
                            continue
                        
                        # Process directory
                        result.append(f"{prefix}{'└── ' if is_last else '├── '}{entry.name}/")
                        _build_tree(entry.path, prefix + ('    ' if is_last else '│   '), depth + 1)
                    elif entry.is_file():
                        # Only include files with specific extensions
                        if any(entry.name.endswith(ext) for ext in included_extensions):
                            result.append(f"{prefix}{'└── ' if is_last else '├── '}{entry.name}")
            
            except PermissionError:
                result.append(f"{prefix}[Permission denied]")
            except Exception as e:
                result.append(f"{prefix}[Error: {str(e)}]")
        
        # Start building the tree from the repository root
        repo_name = os.path.basename(repo_path)
        result.append(f"{repo_name}/")
        _build_tree(repo_path, "")
        
        return "\n".join(result)
    
    def _find_controller_file_with_llm(self, repo_path: str, screen_id: str) -> Optional[Path]:
        """
        Use LLM to find the controller file in the repository based on the screen ID.
        Uses tools for navigation and file exploration.
        
        Args:
            repo_path: Path to the repository root
            screen_id: Screen ID in format "<app>/<area>/<controller>/<action>"
            
        Returns:
            Path to the controller file or None if not found
        """
        
        logger.info(f"Using LLM to find controller file for screen ID: {screen_id}")
        
        # Parse the screen ID
        app_name, area_name, controller_name, action = self._parse_screen_id(screen_id)
        
        # Set up the repository for searching
        setup_repo_search(repo_path)
        
        # Set up a dictionary mapping tool names to their functions
        tools_dict = {
            "navigate_directory": navigate_directory,
            "list_directory": list_directory,
            "read_file": read_file,
            "find_controller": find_controller
        }
        
        # Set up the LLM with tools
        llm_with_tools = self.llm.bind_tools([navigate_directory, list_directory, read_file, find_controller])
        
        # Set up the initial prompt for the LLM
        initial_prompt = f"""You are helping to find a controller file in a BlueJS repository.

You have tools to navigate and explore the repository file system:
- navigate_directory: Change to a directory
- list_directory: List contents of a directory
- read_file: Read the content of a file
- find_controller: Declare you've found the controller file

SCREEN ID:
{screen_id}

The screen ID follows the format "<app>/<area>/<controller>/<action>" where:
- app: The application name ({app_name})
- area: The area or section of the application ({area_name})
- controller: The name of the controller ({controller_name})
- action: The name of the action function within the controller ({action if action else "index"})

WHAT A CONTROLLER FILE LOOKS LIKE:
- Often located in a directory named "controllers" or "controller"
- Usually defined with AMD syntax: define([dependencies], function(dep1, dep2) {{ ... }})
- May have action functions that match the action in the screen ID
- Action functions might be called "index", "show", "edit", etc.
- Functions may receive a "context" parameter which has properties like routeHistory, privateState, state

IMPORTANT SEARCH STRATEGY:
1. You MUST explore directories deeply - controllers are often nested several levels deep
2. Look especially for directories named "js", "app", "src", "frontend", or anything related to {app_name} and {area_name}
3. Always check folders named "controllers", "controller", or any similar variations
4. Don't just stay at the top level - most repositories have organized code in nested subdirectories
5. IMPORTANT: Controllers will NOT be in test/spec folders - skip any test, spec, or __tests__ directories
6. Common patterns to explore:
   - js/controllers/
   - app/{app_name}/{area_name}/controllers/
   - src/controllers/
   - frontend/controllers/
   - {app_name}/controllers/
   - controllers/{app_name}/

7. Skip these directories as they won't contain controllers:
   - node_modules/
   - .git/
   - test/
   - tests/
   - __tests__/
   - spec/
   - specs/
   - e2e/
   - dist/
   - build/

Let's start by looking at the contents of the repository root and then systematically explore promising directories.
"""

        # Set up the conversation with a maximum number of exchanges
        max_attempts = 30  # Increased to 30 for larger repositories
        messages = [HumanMessage(content=initial_prompt)]
        controller_file_path = None
        
        # Track directories we've already explored to avoid repetition
        explored_directories = set(["/"])
        
        # Directories to skip (including test directories)
        skip_directories = {
            "node_modules", ".git", "test", "tests", "__tests__", 
            "spec", "specs", "e2e", "dist", "build"
        }
        
        for attempt in range(max_attempts):
            logger.info(f"LLM search attempt {attempt+1}/{max_attempts}")
            
            # Get response from LLM
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            
            # Process all tool calls and update the conversation
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    tool_call_id = tool_call['id']
                    
                    logger.info(f"Tool call: {tool_name} with args: {tool_args}")
                    
                    # Check if trying to navigate to a test directory
                    if tool_name == 'navigate_directory' and 'path' in tool_args:
                        path = tool_args['path']
                        path_parts = os.path.normpath(path).split(os.sep)
                        
                        # Check if any part of the path is in skip_directories
                        if any(part in skip_directories for part in path_parts):
                            hint_message = f"Skipping directory '{path}' as it appears to be a test/build directory which won't contain controllers."
                            messages.append(HumanMessage(content=hint_message))
                            continue
                        
                        # Check for repeated exploration
                        if path in explored_directories:
                            hint_message = f"You've already explored {path}. Try exploring a different directory or going deeper."
                            messages.append(HumanMessage(content=hint_message))
                            continue
                        
                        explored_directories.add(path)
                    
                    # Get the tool function
                    if tool_name not in tools_dict:
                        error_message = f"Unknown tool: {tool_name}"
                        logger.warning(error_message)
                        messages.append(ToolMessage(content=error_message, tool_call_id=tool_call_id))
                        continue
                    
                    # Execute the tool function with the provided arguments
                    tool_func = tools_dict[tool_name]
                    try:
                        # Call the tool function with the arguments
                        tool_result = tool_func.invoke(tool_args)
                        
                        # Process find_controller tool result
                        if tool_name == 'find_controller':
                            if tool_result.get('success'):
                                logger.info(f"LLM found controller file: {tool_result['file_path']}")
                                logger.info(f"Confidence: {tool_result['confidence']}/10")
                                logger.info(f"Reasoning: {tool_result['reasoning']}")
                                controller_file_path = Path(tool_result['file_path'])
                                return controller_file_path
                            else:
                                logger.warning(f"Controller file declaration failed: {tool_result.get('error')}")
                        
                        # Add guidance for list_directory results
                        if tool_name == 'list_directory':
                            # Check if the result contains "controllers" or "js" directories that should be explored
                            content = str(tool_result)
                            if "controllers" in content.lower() or ("js" in content and "directory" in content.lower()):
                                hint = "\n\nHINT: There appear to be promising directories here that might contain controllers. Consider exploring them."
                                tool_result = str(tool_result) + hint
                        
                        # Log tool result based on tool type
                        if tool_name == 'read_file':
                            logger.info(f"File read: {len(tool_result)} characters")
                            # If the file is very large, truncate the log
                            log_content = tool_result[:200] + "..." if len(tool_result) > 200 else tool_result
                            logger.info(f"File content (truncated): {log_content}")
                        else:
                            logger.info(f"Tool result: {str(tool_result)[:100]}...")
                        
                        # Add the tool result back to the conversation
                        messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call_id))
                        
                    except Exception as e:
                        error_message = f"Error executing tool {tool_name}: {str(e)}"
                        logger.error(error_message)
                        messages.append(ToolMessage(content=error_message, tool_call_id=tool_call_id))
            
            # If we've been searching for a while without finding anything, provide guidance
            if attempt == 5:
                hint_message = """
SEARCH GUIDANCE:
Remember to dive deeper into directories! Controllers are rarely at the top level.
Look specifically for directories named:
1. "controllers" or "controller"
2. Any app-specific directories matching "{app_name}" or "{area_name}"
3. Common web app patterns like "js/", "src/", "app/", etc.

Remember to SKIP all test-related directories as they won't contain actual controllers.

Please be thorough and methodical in your search.
"""
                messages.append(HumanMessage(content=hint_message))
            
            # Additional guidance at attempt 15 to encourage even deeper exploration
            if attempt == 15:
                hint_message = """
CONTINUED SEARCH GUIDANCE:
We're halfway through our search attempts, and we haven't found the controller yet.
Let's consider:
1. Looking for files matching the controller name pattern (with or without .js extension)
2. Exploring any directories we haven't checked yet, especially those with application-specific names
3. Checking for any config files that might give clues about the application structure
4. Remember that controllers won't be in test or build directories

Don't give up! Controllers can be deeply nested or named in unexpected ways.
"""
                messages.append(HumanMessage(content=hint_message))
            
            # Check max attempts
            if attempt >= max_attempts - 1:
                logger.warning(f"LLM could not find controller file after {max_attempts} attempts")
                break
        
        return controller_file_path
    
    def _find_controller_file_with_heuristic(self, repo_path: str, screen_id: str) -> Optional[Path]:
        """
        Find the controller file in the repository based on heuristics.
        
        Args:
            repo_path: Path to the repository root
            screen_id: Screen ID in format "<app>/<area>/<controller>/<action>"
            
        Returns:
            Path to the controller file or None if not found
        """
        logger.info(f"Using heuristic to find controller file for screen ID: {screen_id}")
        
        # Parse the screen ID
        app_name, area_name, controller_name, action = self._parse_screen_id(screen_id)
        
        # The most common pattern is something like: app/{{app_name}}/{{area_name}}/controllers/{{controller_name}}.js
        # But different repositories have different patterns, so we need to try a few.
        repo_path_obj = Path(repo_path)
        
        # Directories to skip - test directories, build dirs, etc.
        skip_dirs = {
            "node_modules", ".git", "test", "tests", "__tests__", 
            "spec", "specs", "e2e", "dist", "build"
        }
        
        # Common patterns to try
        patterns = [
            f"app/{app_name}/{area_name}/controllers/{controller_name}.js",
            f"app/{app_name}/controllers/{controller_name}.js",
            f"js/{app_name}/{area_name}/controllers/{controller_name}.js",
            f"js/{app_name}/controllers/{controller_name}.js",
            f"src/{app_name}/{area_name}/controllers/{controller_name}.js",
            f"src/{app_name}/controllers/{controller_name}.js",
            f"controllers/{controller_name}.js",
            f"*/{controller_name}.js",  # Wild card for one level
            f"*/*/{controller_name}.js",  # Wild card for two levels
            f"*/*/*/{controller_name}.js",  # Wild card for three levels
            f"controllers/**/{controller_name}.js",  # Any level under controllers
            f"**/{controller_name}.js",  # Any level
        ]
        
        # Try direct patterns first (faster)
        for pattern in patterns[:6]:  # Use the specific patterns first
            try:
                file_path = repo_path_obj / pattern
                if file_path.exists():
                    logger.info(f"Found controller file with direct pattern: {pattern}")
                    return file_path
            except Exception as e:
                logger.debug(f"Error checking pattern {pattern}: {e}")
        
        # Now try with Path.glob (slower but more flexible)
        try:
            for pattern in patterns[6:]:
                for file_path in repo_path_obj.glob(pattern):
                    # Check if the file path contains any test directories
                    path_parts = file_path.parts
                    if any(skip_dir in part.lower() for skip_dir in skip_dirs for part in path_parts):
                        logger.debug(f"Skipping test directory: {file_path}")
                        continue
                        
                    logger.info(f"Found controller file with glob pattern: {pattern}")
                    return file_path
        except Exception as e:
            logger.debug(f"Error during glob search: {e}")
        
        # Additionally, try to look for files with the controller name in their name
        try:
            # Recursively walk the directory structure
            for root, dirs, files in os.walk(repo_path):
                # Skip directories that match our skip list
                dirs[:] = [d for d in dirs if d.lower() not in skip_dirs]
                
                # Check files for controller name
                for file in files:
                    if controller_name.lower() in file.lower() and file.endswith('.js'):
                        file_path = Path(os.path.join(root, file))
                        logger.info(f"Found potential controller file by name: {file_path}")
                        return file_path
        except Exception as e:
            logger.debug(f"Error during recursive search: {e}")
        
        logger.warning(f"Could not find controller file for {screen_id} with heuristic")
        return None

    
    def _find_templates(self, repo_path: str, action_code: str, controller_file: Path, action_details: Dict[str, Any]) -> List[Dict[str, str]]:
        """Find templates referenced in the action code and related files."""
        # Start with templates directly referenced in action
        template_references = re.findall(r'template\s*:\s*[\'"]([^\'"]+)[\'"]', action_code)
        
        # Also check templates from action details that the LLM extracted
        if 'templates' in action_details and isinstance(action_details['templates'], list):
            template_references.extend(action_details['templates'])
        
        # Make the list unique
        template_references = list(set(template_references))
        
        template_files = []
        for template_name in template_references:
            # Search for template files with exact and partial matches
            found = False
            for root, _, files in os.walk(repo_path):
                for file in files:
                    # Common template file extensions
                    if file.endswith(('.html', '.ractive', '.mustache', '.hbs', '.handlebars')):
                        # Try exact match with filename
                        if template_name == file or template_name == os.path.splitext(file)[0]:
                            template_path = Path(os.path.join(root, file))
                            with open(template_path, 'r', encoding='utf-8', errors='ignore') as f:
                                template_content = f.read()
                                
                            template_files.append({
                                "name": template_name,
                                "path": str(template_path),
                                "content": template_content,
                                "match_type": "exact"
                            })
                            found = True
                        # Try partial match if exact match failed
                        elif not found and template_name in file:
                            template_path = Path(os.path.join(root, file))
                            with open(template_path, 'r', encoding='utf-8', errors='ignore') as f:
                                template_content = f.read()
                                
                            template_files.append({
                                "name": template_name,
                                "path": str(template_path),
                                "content": template_content,
                                "match_type": "partial"
                            })
                            found = True
        
        # If no templates found, try using AI to identify potential templates
        if not template_files:
            logger.info("No templates found using patterns, trying AI-based identification")
            potential_templates = self._identify_potential_templates(repo_path, controller_file, action_code)
            template_files.extend(potential_templates)
        
        return template_files
    
    def _identify_potential_templates(self, repo_path: str, controller_file: Path, action_code: str) -> List[Dict[str, str]]:
        """Use AI to identify potential template files when pattern matching fails."""
        # Find HTML/template files in the repository
        template_paths = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith(('.html', '.ractive', '.mustache', '.hbs', '.handlebars')):
                    template_paths.append(os.path.join(root, file))
        
        # If too many templates, sample a reasonable number
        if len(template_paths) > 10:
            # Prioritize templates in directories related to the controller
            controller_dir = os.path.dirname(controller_file)
            nearby_templates = [p for p in template_paths if p.startswith(controller_dir)]
            other_templates = [p for p in template_paths if not p.startswith(controller_dir)]
            
            selected_templates = nearby_templates[:5]
            if len(selected_templates) < 5:
                selected_templates.extend(other_templates[:5 - len(selected_templates)])
            template_paths = selected_templates
        
        # If no templates found at all, return empty list
        if not template_paths:
            return []
        
        # Prepare templates for LLM analysis
        template_previews = []
        for path in template_paths:
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(1000)  # Read first 1000 chars as preview
                template_previews.append({
                    "path": path,
                    "preview": content
                })
            except Exception as e:
                logger.warning(f"Error reading template {path}: {str(e)}")
        
        # Use LLM to identify which templates are likely related to the action
        template_identification_prompt = PromptTemplate.from_template(
            """You're analyzing a JavaScript controller action and need to identify which templates it likely uses.
            
            Controller action code:
            ```javascript
            {action_code}
            ```
            
            Available templates in the repository:
            {template_previews}
            
            Based on the action code and template contents, identify which templates are most likely used by this action.
            Return a JSON array with the paths of templates that are most likely related to this action.
            Only include templates that appear to be directly related to this specific action functionality.
            Format: ["path/to/template1", "path/to/template2"]
            """
        )
        
        # Format templates for the prompt
        formatted_previews = "\n\n".join([
            f"Template: {t['path']}\nPreview:\n```html\n{t['preview']}...\n```" 
            for t in template_previews
        ])
        
        try:
            prompt_content = template_identification_prompt.format(
                action_code=action_code,
                template_previews=formatted_previews
            )
            
            structured_llm = self.llm.with_structured_output(list[str])
            identified_templates = structured_llm.invoke(prompt_content)
            
            # Read full content of identified templates
            result = []
            for path in identified_templates:
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        template_content = f.read()
                        
                    result.append({
                        "name": os.path.basename(path),
                        "path": path,
                        "content": template_content,
                        "match_type": "ai_identified"
                    })
                except Exception as e:
                    logger.warning(f"Error reading identified template {path}: {str(e)}")
            
            return result
        except Exception as e:
            logger.error(f"Error in AI template identification: {str(e)}")
            return []
    
    def _convert_templates_to_react(self, templates: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Convert BlueJS templates to React components."""
        if not templates:
            return []
            
        react_components = []
        
        for template in templates:
            try:
                # Create a prompt to convert the template
                conversion_prompt = PromptTemplate.from_template(
                    """Convert this BlueJS template to a React component:
                    
                    Template name: {name}
                    Template path: {path}
                    
                    ```html
                    {content}
                    ```
                    
                    Please convert this to a modern React component:
                    1. Use functional components with hooks
                    2. Handle any mustache/handlebars expressions properly
                    3. Ensure proper prop handling
                    
                    Return just the React component code without any explanations or extra text so we can simply take your output and save it to file.
                    """
                )
                
                message_content = conversion_prompt.format(
                    name=template["name"],
                    path=template["path"],
                    content=template["content"]
                )
                
                # Generate the React component
                response = self.llm.invoke([HumanMessage(content=message_content)])
                
                # Extract code from response
                react_code = response.content
                
                # Remove markdown code block markers if present
                react_code = re.sub(r'^```(?:jsx|javascript|react)?\s*', '', react_code, flags=re.MULTILINE)
                react_code = re.sub(r'\s*```$', '', react_code, flags=re.MULTILINE)
                
                # Save the converted component
                component_name = os.path.splitext(template["name"])[0]
                if not component_name[0].isupper():
                    component_name = component_name.title().replace('-', '').replace('_', '')
                
                react_components.append({
                    "original_template": template["name"],
                    "component_name": component_name,
                    "filename": f"{component_name}.jsx",
                    "content": react_code
                })
                
            except Exception as e:
                logger.error(f"Error converting template {template['name']}: {str(e)}")
                continue
                
        return react_components
    
    def _parse_config_files(self, repo_path: str) -> None:
        """Parse configuration files to extract path aliases."""
        logger.info("Searching for configuration files with path aliases")
        
        # Common config file patterns to search for
        config_patterns = [
            # Webpack config files
            '**/webpack.config.js', '**/webpack.*.config.js', '**/webpack/config*.js',
            # TS/JS config files
            '**/tsconfig.json', '**/jsconfig.json',
            # RequireJS config files
            '**/require.config.js', '**/requirejs.config.js', '**/config.js',
            # Package.json with potential aliases
            '**/package.json',
            # Common BlueJS config patterns (based on what we've seen)
            '**/app.config.js', '**/app.paths.js', '**/paths.js', '**/aliases.js',
        ]
        
        # Track found config files
        found_config_files = []
        
        # Search for config files using patterns
        for pattern in config_patterns:
            for config_path in Path(repo_path).glob(pattern):
                if config_path.is_file():
                    found_config_files.append(config_path)
        
        logger.info(f"Found {len(found_config_files)} potential configuration files")
        
        # Also look for require.config or define calls in any JS file since BlueJS might define aliases anywhere
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.js') and file not in found_config_files:
                    file_path = Path(os.path.join(root, file))
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            # Look for requirejs config patterns
                            if 'require.config(' in content or 'requirejs.config(' in content:
                                found_config_files.append(file_path)
                            # Look for define pattern with a paths object
                            elif 'define(' in content and 'paths' in content and '{' in content:
                                found_config_files.append(file_path)
                    except Exception:
                        continue  # Skip if file can't be read
                    
        # Process each config file
        for config_path in found_config_files:
            try:
                self._extract_aliases_from_config(config_path, repo_path)
            except Exception as e:
                logger.warning(f"Error parsing config file {config_path}: {str(e)}")
        
        # If we found no aliases but did find config files, try harder with AI assistance
        if not self.alias_map and found_config_files:
            logger.info("No aliases found through pattern matching, attempting AI-assisted extraction")
            self._extract_aliases_with_ai(found_config_files, repo_path)
        
        # Fall back to scanning main JS files for alias patterns if we still have no aliases
        if not self.alias_map:
            logger.info("No aliases found in config files, scanning JS files for alias patterns")
            self._scan_for_implicit_aliases(repo_path)
        
        logger.info(f"Found {len(self.alias_map)} path aliases")

    def _extract_aliases_from_config(self, config_path: Path, repo_path: str) -> None:
        """Extract path aliases from a configuration file."""
        file_ext = config_path.suffix.lower()
        
        try:
            with open(config_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Handle different file types differently
            if file_ext == '.json':
                # JSON files (tsconfig.json, jsconfig.json, package.json)
                try:
                    config_data = json.loads(content)
                    
                    # Handle tsconfig/jsconfig.json
                    if 'compilerOptions' in config_data and 'paths' in config_data['compilerOptions']:
                        paths = config_data['compilerOptions']['paths']
                        base_url = config_data.get('compilerOptions', {}).get('baseUrl', '.')
                        base_path = os.path.join(os.path.dirname(config_path), base_url)
                        
                        for alias, paths_list in paths.items():
                            # Remove wildcard endings like /*
                            clean_alias = alias.replace('/*', '')
                            if paths_list and isinstance(paths_list, list):
                                # Take first path and remove wildcard
                                path_value = paths_list[0].replace('/*', '')
                                absolute_path = os.path.normpath(os.path.join(base_path, path_value))
                                self.alias_map[clean_alias] = absolute_path
                
                    # Handle potential aliases in package.json
                    if 'name' in config_data and 'aliasify' in config_data:
                        if 'aliases' in config_data['aliasify']:
                            for alias, path in config_data['aliasify']['aliases'].items():
                                absolute_path = os.path.normpath(os.path.join(os.path.dirname(config_path), path))
                                self.alias_map[alias] = absolute_path
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON in {config_path}")
                
            elif file_ext == '.js':
                # JavaScript files - webpack.config.js, require.config.js, etc.
                
                # Check for webpack resolve.alias pattern
                webpack_alias_pattern = r'(?:resolve\s*:\s*{[^}]*alias\s*:\s*({[^}]*})|alias\s*:\s*({[^}]*}))'
                webpack_matches = re.findall(webpack_alias_pattern, content, re.DOTALL)
                
                for match_group in webpack_matches:
                    for match in match_group:
                        if not match:
                            continue
                        
                        # Extract key-value pairs from the alias object
                        alias_entries = re.findall(r'[\'"]([@\w\-./]+)[\'"]\s*:\s*[\'"]([^\'"]*)[\'"]\s*,?', match)
                        for alias, path in alias_entries:
                            # Make the path absolute
                            if path.startswith('./') or path.startswith('../'):
                                absolute_path = os.path.normpath(os.path.join(os.path.dirname(config_path), path))
                            else:
                                absolute_path = os.path.join(repo_path, path)
                            
                            self.alias_map[alias] = absolute_path
                
                # Check for requirejs/BlueJS paths pattern
                require_paths_pattern = r'(?:require\.config|requirejs\.config|define)\(\s*{[^}]*paths\s*:\s*({[^}]*})'
                require_matches = re.findall(require_paths_pattern, content, re.DOTALL)
                
                for match in require_matches:
                    # Extract key-value pairs from the paths object
                    path_entries = re.findall(r'[\'"]([\w\-./]+)[\'"]\s*:\s*[\'"]([^\'"]*)[\'"]\s*,?', match)
                    for alias, path in path_entries:
                        # Make the path absolute, handling relative paths
                        if path.startswith('./') or path.startswith('../'):
                            absolute_path = os.path.normpath(os.path.join(os.path.dirname(config_path), path))
                        else:
                            # For non-relative paths in requirejs, we need to guess the base directory
                            # Common base directories for requirejs
                            for base_dir in ['js', 'src', 'app', '.']:
                                potential_path = os.path.join(repo_path, base_dir, path)
                                if os.path.exists(potential_path) or os.path.exists(potential_path + '.js'):
                                    absolute_path = potential_path
                                    break
                            else:
                                # Default to repo root if no match found
                                absolute_path = os.path.join(repo_path, path)
                        
                        self.alias_map[alias] = absolute_path
        
        except Exception as e:
            logger.warning(f"Error processing config file {config_path}: {str(e)}")

    def _extract_aliases_with_ai(self, config_files: List[Path], repo_path: str) -> None:
        """Use AI to extract aliases from config files when pattern matching fails."""
        # Prepare config files content for AI analysis
        config_contents = []
        for config_path in config_files[:5]:  # Limit to 5 files to avoid token limits
            try:
                with open(config_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                config_contents.append({
                    "path": str(config_path),
                    "content": content
                })
            except Exception as e:
                logger.warning(f"Could not read config file {config_path}: {str(e)}")
        
        if not config_contents:
            return
        
        # Prepare prompt for AI
        alias_extraction_prompt = PromptTemplate.from_template(
            """You are analyzing configuration files from a JavaScript project to find path aliases or module mappings.
            
            Here are the configuration files:
            {config_files}
            
            Look for any path aliases, module mappings, or module resolution configurations.
            These might appear as:
            1. In webpack configs: resolve.alias
            2. In TS/JS configs: compilerOptions.paths
            3. In RequireJS configs: paths
            4. Custom BlueJS configurations for module resolution
            
            Return a JSON object mapping alias names to their actual paths. For example:
            {{"@app": "./src/app", "@common": "./src/common", "common": "./src/common"}}
            
            Make sure to adjust relative paths to be relative to the config file location.
            If no aliases are found, return an empty object {{}}.
            """
        )
        
        # Format config files for prompt
        config_files_text = "\n\n".join([
            f"File: {c['path']}\n```\n{c['content'][:2000]}...\n```" 
            for c in config_contents
        ])
        
        try:
            prompt_content = alias_extraction_prompt.format(config_files=config_files_text)
            structured_llm = self.llm.with_structured_output(dict[str, str])
            alias_result = structured_llm.invoke(prompt_content)
            
            # Process and add the aliases
            for alias, path in alias_result.items():
                # Try to find which config file this came from to resolve relative paths
                most_likely_config = config_contents[0]['path']  # Default to first
                for config in config_contents:
                    if path in config['content'] or alias in config['content']:
                        most_likely_config = config['path']
                        break
                        
                # Make the path absolute
                if path.startswith('./') or path.startswith('../'):
                    absolute_path = os.path.normpath(os.path.join(os.path.dirname(most_likely_config), path))
                else:
                    absolute_path = os.path.join(repo_path, path)
                    
                self.alias_map[alias] = absolute_path
                
        except Exception as e:
            logger.error(f"Error in AI alias extraction: {str(e)}")

    def _scan_for_implicit_aliases(self, repo_path: str) -> None:
        """Scan JS files for implicit alias definitions when no explicit config is found."""
        # Common patterns for directory structure that often correspond to aliases
        common_dirs = ['src', 'app', 'lib', 'components', 'modules']
        
        for common_dir in common_dirs:
            dir_path = os.path.join(repo_path, common_dir)
            if os.path.isdir(dir_path):
                # Add the common directory itself as a potential alias
                self.alias_map[common_dir] = dir_path
                
                # Check for subdirectories that might be aliases
                try:
                    with os.scandir(dir_path) as it:
                        for entry in it:
                            if entry.is_dir() and not entry.name.startswith('.'):
                                # Add subdirectory as potential alias
                                self.alias_map[entry.name] = entry.path
                                # Also add with parent directory prefix
                                self.alias_map[f"{common_dir}/{entry.name}"] = entry.path
                except Exception:
                    continue
        
        # Look for common alias patterns in import statements
        alias_patterns = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.js'):
                    try:
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                        # Look for import/require statements with patterns that look like aliases
                        import_patterns = re.findall(r'(?:require\([\'"]|from\s+[\'"])([^\'".]+)[\'"]', content)
                        for pattern in import_patterns:
                            if '/' in pattern and not pattern.startswith('.'):
                                # This looks like it could be an aliased import
                                parts = pattern.split('/')
                                potential_alias = parts[0]
                                
                                if potential_alias not in alias_patterns:
                                    alias_patterns.append(potential_alias)
                    except Exception:
                        continue
        
        # Try to resolve these potential aliases
        for alias in alias_patterns:
            # Look for directories with matching names
            for root, dirs, _ in os.walk(repo_path):
                if alias in dirs:
                    self.alias_map[alias] = os.path.join(root, alias)
                    break
        
        logger.info(f"Added {len(self.alias_map)} potential implicit aliases")
    
    def _analyze_file_dependencies(self, file_path: Path, repo_path: str) -> Dict[str, Any]:
        """
        Recursively analyze a file and its dependencies.
        
        Args:
            file_path: Path to the file to analyze
            repo_path: Path to the repository root
            depth: Current recursion depth to prevent too deep analysis
            
        Returns:
            Dictionary with file analysis information
        """

        resolved_dependencies = {}
            
        if file_path in self.visited_files:
            logger.info(f"Already visited {file_path}")
            return {"message": "Already visited"}
            
        self.visited_files.add(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {str(e)}")
            return {"error": f"Could not read file {file_path}: {str(e)}"}
            
        # Extract define/require dependencies
        logger.info(f"Analyzing {file_path}")
        dependencies = self._extract_dependencies(content, file_path)
        self.dependency_graph[str(file_path)] = dependencies
        
        # Use LLM to analyze the file content
        analysis_prompt = PromptTemplate.from_template(
            """Convert this file to a set of requirements/psuedo code.
            If any @octagon packages/libraries are used, recommend retaining them as they are compatible with react
            If any public npm packages are used, recommend retaining those as well for re-useability.
            
            ONLY REPLY WITH THE REQUIREMENTS. DO NOT ADD ANY OTHER INFORMATION OR COMMENTS.

            File path: {file_path}
            
            ```javascript
            {content}
            ```
            """
        )
        
        retry_count = 3

        while retry_count > 0:
            try:
                message_content = analysis_prompt.format(file_path=str(file_path), content=content)
                response = self.llm.invoke([HumanMessage(content=message_content)])
                
                analysis = response.content

                logger.info(f"Analysis: {analysis}")
                
                # Save analysis to memory
                self.memory.save_context(
                    {"input": f"Analyzed file: {file_path}"},
                    {"output": analysis}
                )
                break
            except Exception as e:
                retry_count -= 1
                if retry_count > 0:
                    logger.warning(f"Could not analyze file {file_path}: {str(e)}. Retrying...")
                else:
                    logger.warning(f"Could not analyze file {file_path}: {str(e)}")
                    return {"error": f"Could not analyze file {file_path}: {str(e)}"}
        
        # If there are dependencies, let the LLM prioritize which ones to analyze next
        dependency_analysis = {}
        if dependencies:
            # Resolve all dependency paths
            for dep in dependencies:
                dep_path = self._resolve_dependency_path(dep, file_path, repo_path)
                if dep_path and os.path.exists(dep_path):
                    # Get a preview of the dependency content
                    try:
                        with open(dep_path, 'r', encoding='utf-8', errors='ignore') as f:
                            preview = f.read()
                        resolved_dependencies[dep] = {
                            "path": dep_path,
                            "preview": preview
                        }
                    except Exception:
                        pass  # Skip if we can't read the file
            
        if resolved_dependencies:
            # Analyze prioritized dependencies
            for dep in resolved_dependencies:
                dep_path = resolved_dependencies[dep]["path"]
                dependency_analysis[dep] = self._analyze_file_dependencies(
                    Path(dep_path), repo_path
                )
        
        return {
            "path": str(file_path),
            "dependencies": dependencies,
            "analysis": analysis,
            "dependency_analysis": dependency_analysis
        }
    
    def _extract_dependencies(self, content: str, file_path: Path) -> List[str]:
        """Extract define/require dependencies from file content."""
        # Match define(['dep1', 'dep2', ...], function) pattern
        define_pattern = r"define\(\s*\[\s*((?:['\"]\w+['\"],?\s*)+)\]"
        # Match require('dep') pattern
        require_pattern = r"require\(\s*['\"]([^'\"]+)['\"]\s*\)"
        
        dependencies = []
        
        # Extract define dependencies
        define_matches = re.findall(define_pattern, content)
        for match in define_matches:
            # Extract individual dependencies
            dep_matches = re.findall(r"['\"]([^'\"]+)['\"]", match)
            dependencies.extend(dep_matches)
            
        # Extract require dependencies
        require_matches = re.findall(require_pattern, content)
        dependencies.extend(require_matches)
        
        return list(set(dependencies))  # Remove duplicates
    
    def _resolve_dependency_path(self, dependency: str, current_file: Path, repo_path: str) -> Optional[str]:
        """Resolve a dependency string to an actual file path."""
        # Check if this is an aliased path
        for alias, alias_path in self.alias_map.items():
            if dependency.startswith(alias):
                # Replace alias with actual path
                relative_path = dependency[len(alias):].lstrip('/')
                # Try different extensions
                for ext in ['.js', '.jsx', '.ts', '.tsx']:
                    full_path = os.path.normpath(os.path.join(alias_path, relative_path + ext))
                    if os.path.exists(full_path):
                        return full_path
                
                # Check for index files in directory
                dir_path = os.path.join(alias_path, relative_path)
                if os.path.isdir(dir_path):
                    for index_file in ['index.js', 'index.jsx', 'index.ts', 'index.tsx']:
                        full_path = os.path.join(dir_path, index_file)
                        if os.path.exists(full_path):
                            return full_path
                
                # Default to .js if nothing else found
                return os.path.join(alias_path, relative_path) + '.js'
                
        # Relative path from current file
        if dependency.startswith('./') or dependency.startswith('../'):
            base_path = os.path.normpath(os.path.join(os.path.dirname(current_file), dependency))
            
            # Try different extensions
            for ext in ['.js', '.jsx', '.ts', '.tsx']:
                if os.path.exists(base_path + ext):
                    return base_path + ext
            
            # Check for index files in directory
            if os.path.isdir(base_path):
                for index_file in ['index.js', 'index.jsx', 'index.ts', 'index.tsx']:
                    full_path = os.path.join(base_path, index_file)
                    if os.path.exists(full_path):
                        return full_path
            
            # Default to .js
            return base_path + '.js'
            
        # Try to find in common locations
        common_paths = [
            os.path.join(repo_path, 'src', dependency),
            os.path.join(repo_path, 'app', dependency),
            os.path.join(repo_path, 'lib', dependency),
            os.path.join(repo_path, 'vendor', dependency),
            os.path.join(repo_path, 'node_modules', dependency)
        ]
        
        # Try different extensions and index files
        for base_path in common_paths:
            # Try with extensions
            for ext in ['.js', '.jsx', '.ts', '.tsx']:
                if os.path.exists(base_path + ext):
                    return base_path + ext
            
            # Try index files
            if os.path.isdir(base_path):
                for index_file in ['index.js', 'index.jsx', 'index.ts', 'index.tsx']:
                    full_path = os.path.join(base_path, index_file)
                    if os.path.exists(full_path):
                        return full_path
        
        logger.warning(f"Could not resolve dependency path: {dependency}")
        return None
            
    def _generate_requirements_with_context(
        self, 
        repo_path: str, 
        controller_file: Path,
        file_tree: str,
        controller_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate requirements with comprehensive context from dependency analysis."""
        
        # Read the controller file
        with open(controller_file, 'r', encoding='utf-8', errors='ignore') as f:
            controller_content = f.read()
            
        # Prepare dependency information
        dependency_info = json.dumps(self.dependency_graph, indent=2)
        
        # Get conversation history from memory
        conversation_history = self.memory.load_memory_variables({})
        analysis_context = conversation_history.get("history", "No previous analysis available.")
        
        # Find config files for path aliases
        config_files = []
        for alias, path in self.alias_map.items():
            config_files.append({
                "alias": alias,
                "path": path
            })
        
        requirements_prompt = PromptTemplate.from_template(
            """You are examining code from a JavaScript application that uses a custom framework called BlueJS. 
            Please analyze the code patterns and structure to help translate this application to React.
            
            File tree:
            ```
            {file_tree}
            ```
            
            Controller file: {controller_file}
            ```javascript
            {controller_content}
            ```
            
            Dependency Analysis:
            {dependency_analysis}
            
            Previous file analysis:
            {analysis_context}
            
            Path aliases:
            {config_info}
            
            Based solely on the code patterns you observe (not on prior knowledge of BlueJS), create a requirements document with these sections:
            1. Application Overview - What appears to be the purpose of this application based on the code?
            2. Data Flow and State Management - How does the application manage data? What state management would be needed in React?
            3. UI Components - What UI components will need to be created in React based on the templates?
            4. API Integrations - What external services or APIs does the application interact with?
            5. Routing Structure - How does routing appear to work in this application? What pages/urls exist?
            6. React Implementation Approach - How would you recommend implementing this in React?
            
            Focus on concrete patterns you observe in the code without making assumptions about how BlueJS works internally.
            The requirements should be detailed enough to completely re-write the application without loss of functionality.
            """
        )
        
        config_info = "\n".join([
            f"- {c['alias']} -> {c['path']}"
            for c in config_files
        ]) if config_files else "No path aliases found."
        
        message_content = requirements_prompt.format(
            file_tree=file_tree,
            controller_file=str(controller_file),
            controller_content=controller_content,
            dependency_analysis=json.dumps(controller_analysis, indent=2),
            analysis_context=analysis_context,
            config_info=config_info
        )
        
        requirements = self.llm.invoke([HumanMessage(content=message_content)]).content
        
        return {
            "full_requirements": requirements
        }

    def find_all_view_files(self, repo_path: str) -> List[Path]:
        """
        Find all view files in the repository by locating 'view' or 'views' directories.
        
        Args:
            repo_path: Path to the repository root
            
        Returns:
            List of paths to view files
        """
        logger.info(f"Finding all view files in {repo_path}")
        view_files = []
        view_dir_names = ['view', 'views']
        
        # Walk through the repository to find view directories
        for root, dirs, files in os.walk(repo_path):
            # Skip certain directories that won't contain views
            dirs[:] = [d for d in dirs if d.lower() not in [
                'node_modules', '.git', 'test', 'tests', '__tests__', 
                'spec', 'specs', 'e2e', 'dist', 'build'
            ]]
            
            # Check if current directory is a view directory
            current_dir = os.path.basename(root).lower()
            is_view_dir = current_dir in view_dir_names
            
            # Also check if any parent directory in the path is a view directory
            path_parts = Path(root).relative_to(repo_path).parts
            has_view_parent = any(part.lower() in view_dir_names for part in path_parts)
            
            if is_view_dir or has_view_parent:
                # Add all JavaScript/TypeScript files in this directory
                for file in files:
                    if file.endswith(('.js', '.jsx', '.ts', '.tsx', '.hbs', '.handlebars', '.mustache', '.html')):
                        view_file_path = Path(os.path.join(root, file))
                        view_files.append(view_file_path)
                        logger.info(f"Found view file: {view_file_path}")
        
        logger.info(f"Found {len(view_files)} view files in total")
        return view_files

    def analyze_view_dependencies(self, view_file: Path, repo_path: str) -> List[Path]:
        """
        Analyze a view file to find all its dependencies (imports/requires and templates).
        
        Args:
            view_file: Path to the view file
            repo_path: Path to the repository root
            
        Returns:
            List of paths to dependent files
        """
        logger.info(f"Analyzing dependencies for view file: {view_file}")
        dependencies = set()
        dependencies.add(view_file)  # Include the original view file
        
        # Reset visited files for this analysis
        self.visited_files = set()
        
        # Start recursive dependency analysis
        self._analyze_file_dependencies_recursive(view_file, repo_path, dependencies)
        
        logger.info(f"Found {len(dependencies)} dependencies for {view_file}")
        return list(dependencies)

    def _analyze_file_dependencies_recursive(self, file_path: Path, repo_path: str, dependencies: Set[Path]):
        """
        Recursively analyze a file to find its dependencies.
        
        Args:
            file_path: Path to the file to analyze
            repo_path: Path to the repository root
            dependencies: Set to collect dependent files
        """
        if file_path in self.visited_files:
            return
        
        self.visited_files.add(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Find all imports and requires
            import_patterns = [
                r'import\s+[\w\{\},\s*]+\s+from\s+[\'"]([^\'"]+)[\'"]',  # ES6 imports
                r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',  # CommonJS requires
                r'define\s*\(\s*\[[^\]]*[\'"]([^\'"]+)[\'"][^\]]*\]',  # AMD define dependencies
                r'@import\s+[\'"]([^\'"]+)[\'"]',  # CSS imports
                r'templateUrl\s*:\s*[\'"]([^\'"]+)[\'"]',  # Template URLs
                r'<include\s+src=[\'"]([^\'"]+)[\'"]',  # HTML includes
                r'<link[^>]*href=[\'"]([^\'"]+)[\'"]',  # CSS links
                r'<script[^>]*src=[\'"]([^\'"]+)[\'"]',  # Script tags
            ]
            
            for pattern in import_patterns:
                for match in re.finditer(pattern, content):
                    imported_path = match.group(1)
                    
                    # Skip external modules and absolute paths
                    if (imported_path.startswith('http') or 
                        imported_path.startswith('/') or 
                        imported_path.startswith('@angular') or
                        not '/' in imported_path):
                        continue
                    
                    # Resolve relative imports
                    full_path = self._resolve_import_path(imported_path, file_path, repo_path)
                    if full_path and full_path.exists():
                        logger.debug(f"Found dependency: {full_path}")
                        dependencies.add(full_path)
                        
                        # Recursively analyze this dependency
                        self._analyze_file_dependencies_recursive(full_path, repo_path, dependencies)
        
        except Exception as e:
            logger.error(f"Error analyzing dependencies for {file_path}: {str(e)}")

    def _resolve_import_path(self, import_path: str, current_file: Path, repo_path: str) -> Optional[Path]:
        """
        Resolve an import path to an absolute file path.
        
        Args:
            import_path: The import path from the source code
            current_file: The file containing the import
            repo_path: Path to the repository root
            
        Returns:
            Resolved absolute file path or None if not found
        """
        # Handle alias paths first (e.g., @app/component)
        if import_path.startswith('@') and '/' in import_path:
            for alias, alias_path in self.alias_map.items():
                if import_path.startswith(alias):
                    relative_path = import_path[len(alias):].lstrip('/')
                    return Path(os.path.join(repo_path, alias_path, relative_path))
        
        # Handle relative paths
        if import_path.startswith('./') or import_path.startswith('../'):
            base_dir = current_file.parent
            relative_path = os.path.normpath(os.path.join(base_dir, import_path))
            return Path(relative_path)
        
        # Handle absolute paths relative to repo root
        return Path(os.path.join(repo_path, import_path))

    def convert_views_to_react(self, repo_url: str) -> Dict[str, Any]:
        """
        Find all view files and convert them to React components.
        
        Args:
            repo_path: Path to the repository root
            
        Returns:
            Dictionary containing conversion results
        """
        logger.info(f"Starting view-to-React conversion for {repo_url}")
        self.start_time = time.time()

        # Ensure the repository storage directory exists
        os.makedirs(REPO_STORAGE_DIR, exist_ok=True)
        
        # Clone the repository to our dedicated storage location
        repo_path = self._clone_repo(repo_url, REPO_STORAGE_DIR)
        
        # Find all view files
        view_files = self.find_all_view_files(repo_path)
        
        # Parse configuration files for path aliases
        self._parse_config_files(repo_path)
        logger.info(f"Parsed configuration files for path aliases. Found {len(self.alias_map)} aliases.")
        
        # Process each view file
        conversion_results = {}
        for view_file in view_files:
            try:
                logger.info(f"Processing view file: {view_file}")
                
                # Analyze dependencies for this view
                dependencies = self.analyze_view_dependencies(view_file, repo_path)
                
                # Read all dependent files
                dependent_files_content = {}
                for dep_file in dependencies:
                    try:
                        with open(dep_file, 'r', encoding='utf-8', errors='ignore') as f:
                            file_content = f.read()
                        relative_path = os.path.relpath(dep_file, repo_path)
                        dependent_files_content[relative_path] = file_content
                    except Exception as e:
                        logger.error(f"Error reading dependent file {dep_file}: {str(e)}")
                
                # Convert to React using LLM
                react_components = self._convert_to_react(view_file, dependent_files_content)
                
                # Store results
                relative_view_path = os.path.relpath(view_file, repo_path)
                conversion_results[relative_view_path] = {
                    "original_view": str(view_file),
                    "dependencies": [str(dep) for dep in dependencies],
                    "react_components": react_components
                }
                
            except Exception as e:
                logger.error(f"Error processing view file {view_file}: {str(e)}")
                conversion_results[str(view_file)] = {
                    "error": str(e)
                }
        
        elapsed_time = time.time() - self.start_time
        logger.info(f"View-to-React conversion completed in {elapsed_time:.2f} seconds")
        
        return {
            "conversion_results": conversion_results,
            "view_count": len(view_files),
            "analysis_time_seconds": elapsed_time
        }

    def _convert_to_react(self, view_file: Path, dependent_files_content: Dict[str, str]) -> Dict[str, Any]:
        """
        Use LLM to convert a view file and its dependencies to React components with structured output.
        
        Args:
            view_file: Path to the view file
            dependent_files_content: Dictionary mapping file paths to their content
            
        Returns:
            Dictionary containing React components
        """
        logger.info(f"Converting {view_file} to React components")
        
        # Import pydantic models for structured output
        from pydantic import BaseModel, Field
        from typing import List, Optional as OptionalType
        
        # Define the structured output schema
        class File(BaseModel):
            filename: str = Field(description="Filename for the component (e.g., 'Component.tsx', 'utils/helpers.ts')")
            content: str = Field(description="Complete source code for the component")
        
        class ReactConversionFiles(BaseModel):
            files: List[File] = Field(description="List of React component files")
        
        # Configure the LLM for structured output
        structured_llm = self.llm.with_structured_output(ReactConversionFiles)
        
        # Prepare the context for the LLM
        system_prompt = """You are an expert at converting legacy web applications to modern React TypeScript applications.
Your task is to convert view files and their dependencies to React components.

Guidelines:
- Convert view context variables to props
- Convert model data to state
- Convert templates to React JSX components
- For unrecognized web components, import them from '@mds/react' with PascalCase names
- Props for MDS React components should be camelCase
- Place utility functions in appropriate files inside a utils folder
- Convert require('@octagon/...') to import statements
- Return a structured collection of files that together implement the view in React
- Mark the main component file as the entrypoint
"""
        
        user_prompt = """Here are the files involved in this view that need to be converted to React:
"""
        
        # Add file contents to the prompt
        for file_path, content in dependent_files_content.items():
            user_prompt += f"\n\nFile: {file_path}\n```\n{content}\n```"
        
        # Call LLM to convert to React with structured output
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = structured_llm.invoke(messages)
            
            # Convert the structured output to our standard format
            result = {
                "components": {}
            }
            
            for file in response.files:
                result["components"][file.filename] = {
                    "content": file.content,
                    "description": file.description or "",
                    "entrypoint": file.entrypoint
                }
                
            if response.explanation:
                result["explanation"] = response.explanation
                
            return result
            
        except Exception as e:
            logger.error(f"Error converting {view_file} to React: {str(e)}")
            return {"error": str(e)}


def convert_blueJS_repo(repo_url: str) -> Dict[str, Any]:
    """
    Analyze a BlueJS repository to extract requirements and convert to React.
    
    Args:
        repo_url: URL of the git repository to clone and analyze
        screen_id: Path-like string in format "<app_name>/<area_name>/<controller_file_name>/<action>"
        
    Returns:
        Dictionary containing analysis results, requirements and converted React components
    """
    analyzer = CodeAnalyzer()
    return analyzer.convert_views_to_react(repo_url)


if __name__ == "__main__":
    # Import required modules for the main function
    from dotenv import load_dotenv
    import os
    import sys
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get repo_url and screen_id from environment variables
    repo_url = os.getenv("REPO_URL")
    
    # Validate that we have the required environment variables
    if not repo_url:
        print("Error: REPO_URL must be set in .env file")
        sys.exit(1)
    
    # Run the analysis
    result = convert_blueJS_repo(repo_url)
    print(result)

