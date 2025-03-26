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
import rapidjson # faster is better (also it handles poorly-formatted JSON)
import time
import logging
import hashlib
import glob

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback
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

# Resolved path cache
RESOLVED_PATH_CACHE = {}

# Paths that cannot be resolved
UNRESOLVABLE_PATHS = set()

# Misc metrics
GENERATED_FILE_COUNT = 0

class CodeAnalyzer:
    """Analyzes BlueJS code repositories to extract requirements and convert to React."""
    
    def __init__(self, model_name: str = "gpt-4o", timeout_minutes: int = 30):
        """Initialize with the specified model."""
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
        # Track converted files to enable reuse
        self.converted_files = {
            "components": {},  # Maps source file path -> generated component files
            "utils": {},       # Maps utility name -> generated utility file path
            "settings": {},    # Maps settings name -> generated settings file path
            "types": {},       # Maps type name -> generated type file path
        }
        # The single output directory for the entire project
        self.project_output_dir = None
            
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
            '.js', '.jsx', '.ts', '.tsx', '.json', '.html',
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
    
    def _find_manifest_and_registry_files(self, repo_path: str) -> List[str]:
        """
        Finds manifest.js and registry.js files in the provided folder.
        """

        # Search for files named "manifest.js" or "registry.js" under the given directory
        search_patterns = ["**/src/**/manifest.js", "**/src/**/registry.js"]
        found_files = []

        for pattern in search_patterns:
            found_files.extend(glob.glob(os.path.join(repo_path, pattern), recursive=True))
    
        return found_files

    def _remove_comments(self, js_code: str) -> str:
        """
        Removes comments from a JavaScript code string.

        Args:
            js_code: The JavaScript code string to remove comments from.

        Returns:
            The JavaScript code string with comments removed.
        """
        # Remove single-line comments
        js_code = re.sub(r'//.*', '', js_code)
        # Remove multi-line comments
        js_code = re.sub(r'/\*[\s\S]*?\*/', '', js_code)
        return js_code

    def _extract_json_from_javascript(self, file_path: str) -> Dict[str, Any]:
        """
        Converts a simple JavaScript AMD module to JSON.

        Args:
            file_path: The path to the JavaScript file.
            
        Returns:
            A Dictionary representation of the module, or None if conversion fails.
        """

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            js_code = f.read()

        # Attempt to locate just the JSON portion of the file, navigating around the
        # "define({ ... })" of an AMD module.
        match = re.search(r"define\(\s*.*?,\s*function\s*\(\)\s*\{(.*)\}\s*\)", js_code, re.DOTALL)
        if not match:
            match = re.search(r"define\(\s*({.*?})\s*\)", js_code, re.DOTALL)
        if match:
            try:
                module_content = match.group(1).strip()

                # Add double quotes around keys. Do this globally, to the entire module_content string.
                module_content = re.sub(r'(\w+):', r'"\1":', module_content)
                
                # Wrap boolean values with double quotes
                module_content = re.sub(r'\btrue\b', '"true"', module_content)
                module_content = re.sub(r'\bfalse\b', '"false"', module_content)

                # Convert single quotes to double quotes
                module_content = re.sub(r"'", '"', module_content)
                
                # Remove JavaScript comments, which are invalid in JSON files.
                module_content = self._remove_comments(module_content)

                # Return a dictionary representation of the module, or None if conversion fails.
                return rapidjson.loads(module_content,
                                       parse_mode=rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS)

            except Exception as e:
                print(f"Error converting " + file_path + " to JSON: " + str(e))
                print(f"Here is the offending JSON: " + module_content)
                return None
        else:
            print("No AMD module definition found.")
            return None

    def _load_json_maps_from_files(self, repo_path: str) -> Dict[str, Any]:
        """
        Loads JSON maps from files in the given directory.
        """
        files = self._find_manifest_and_registry_files(repo_path)
        json_maps = {}

        for file in files:
            json_map = self._extract_json_from_javascript(file)
            if json_map:
                json_maps[file] = json_map
        
        return json_maps

    def _build_controller_to_component_map(self, repo_path: str) -> Dict[str, Set[str]]:
        """ 
        Builds a Dictionary mapping controllers to components. 
        """
        logger.info(f"Locating manifest and registry files in {repo_path}.")
        try:
            registry_maps = self._load_json_maps_from_files(repo_path)
        except Exception as e:
            print(f"Error building controller to component map: {e}")
            return None

        controllers_with_components = {}
            
        # Create a Dictionary mapping controllers to a Set of components.
        def get_controllers_with_components(registry_map: Dict[str, Any]) -> Dict[str, Set[str]]:
            """
            Given a map holding registry data, return a Dictionary mapping controllers to a Set of components.
            """            
            area_data = registry_map["area"]
            if "controllers" in area_data:
                controller_data = area_data["controllers"]
                for controller_name, controller_data in controller_data.items():
                    if "components" in controller_data:
                        for component_name, component_data in controller_data["components"].items():
                            if controller_name not in controllers_with_components:
                                controllers_with_components[controller_name] = set()
                            controllers_with_components[controller_name].add(component_name)

        for registry_map in registry_maps:
            controller_map = get_controllers_with_components(registry_maps[registry_map])
            
            # It's possible controller_map is empty, as some manifest/registry files do not include controller information.
            if controller_map is not None:
                controllers_with_components.update(controller_map)
        
        logger.info(f"Collected {len(controllers_with_components)} controllers with components.")
        return controllers_with_components

    def _build_component_to_controller_map(self, controller_to_component_map):
        """
        Utility function to invert the controller-to-component map,
        producing a component-to-controller map.
        """
        if controller_to_component_map is None:
            return None
        
        component_to_controller_map = {}
        for controller, components in controller_to_component_map.items():
            for component in components:
                component_to_controller_map[component] = controller
        return component_to_controller_map


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
            '**/app.config.js', '**/app.paths.js', '**/paths.js', '**/aliases.js', "**/registery.js", "**/main.js"
        ]

        # Track found config files
        found_config_files = []

        # Search for config files using patterns
        for pattern in config_patterns:
            for config_path in glob.glob(str(Path(repo_path) / pattern), recursive=True):
                logger.info(f"Checking file: {config_path}")
                if Path(config_path).is_file():
                    found_config_files.append(Path(config_path))

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

        # Check if we've already deemed this un-resolvable
        if dependency in UNRESOLVABLE_PATHS:
            return None

        # Check if this is an aliased path
        logger.info(f"Resolving dependency: {dependency} in repo {repo_path}")

        # logger.info(f"Resolving dependency: {dependency}")
        for alias, alias_path in self.alias_map.items():
            if dependency.startswith(alias):
                # Replace alias with actual path
                relative_path = dependency[len(alias):].lstrip('/')
                # Try different extensions
                for ext in ['.js', '.jsx', '.ts', '.tsx', '.json', '.html', '.css', '.scss', '.less']:
                    full_path = os.path.normpath(os.path.join(alias_path, relative_path + ext))
                    if os.path.exists(full_path):
                        return full_path
                
                # Check for index files in directory
                dir_path = os.path.join(alias_path, relative_path)
                if os.path.isdir(dir_path):
                    for index_file in ['index.js', 'index.jsx', 'index.ts', 'index.tsx', 'index.json', 'index.html', 'index.css', 'index.scss', 'index.less']:
                        full_path = os.path.join(dir_path, index_file)
                        if os.path.exists(full_path):
                            return full_path
                
        # Relative path from current file
        if dependency.startswith('./') or dependency.startswith('../'):
            base_path = os.path.normpath(os.path.join(os.path.dirname(current_file), dependency))
            
            # Try different extensions
            for ext in ['.js', '.jsx', '.ts', '.tsx', '.json', '.html', '.css', '.scss', '.less']:
                if os.path.exists(base_path + ext):
                    return base_path + ext
            
            # Check for index files in directory
            if os.path.isdir(base_path):
                for index_file in ['index.js', 'index.jsx', 'index.ts', 'index.tsx', 'index.json', 'index.html', 'index.css', 'index.scss', 'index.less']:
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
            for ext in ['.js', '.jsx', '.ts', '.tsx', '.json', '.html', '.css', '.scss', '.less']:
                # logger.info(f"Trying path: {base_path + ext}") # DEBUG
                if os.path.exists(base_path + ext):
                    return base_path + ext
            
            # Try index files
            if os.path.isdir(base_path):
                for index_file in ['index.js', 'index.jsx', 'index.ts', 'index.tsx', 'index.json', 'index.html', 'index.css', 'index.scss', 'index.less']:
                    full_path = os.path.join(base_path, index_file)
                    if os.path.exists(full_path):
                        return full_path
        
        logger.warning(f"Could not resolve dependency path: {dependency}")
        # Add this dependency to the UNRESOLVABLE_PATHS set.
        UNRESOLVABLE_PATHS.add(dependency)
        return None

    def find_defined_controller_files(self, controller_names: List[str], repo_path: str) -> List[Path]:
        """
        Find all controller files in the repository that match the given keys.
        
        Args:
            controller_names: List of controller names to find in filesystem
            repo_path: Path to the repository root
            
        Returns:
            List of paths to controller files
        """
        logger.info(f"Finding all controller files in {repo_path}")
        controller_dir_names = ['controller', 'controllers']
        controller_files = []
        discovered_controller_names = []
        unexpected_controller_names = []

        matched_controller_files = [] # List of controllers that matched provided controller_names list
        
        # Walk through the repository to find controller directory
        for root, dirs, files in os.walk(repo_path):
            # Skip certain directories that won't contain views
            dirs[:] = [d for d in dirs if d.lower() not in [
                'node_modules', '.git', 'test', 'tests', '__tests__', 
                'spec', 'specs', 'e2e', 'dist', 'build'
            ]]
            
            # Check if current directory is a controller directory
            current_dir = os.path.basename(root).lower()
            is_controller_dir = current_dir in controller_dir_names
            
            # Also check if any parent directory in the path is a controller directory
            path_parts = Path(root).relative_to(repo_path).parts
            has_controller_parent = any(part.lower() in controller_dir_names for part in path_parts)
            
            if is_controller_dir or has_controller_parent:
                # Add all JavaScript/TypeScript files in this directory
                for file in files:
                    if file.endswith(('.js', '.ts', '.jsx', '.tsx')):
                        # Get just the prefix of the file
                        file_prefix = file[:file.rindex('.')]
                        if file_prefix not in controller_names:
                            logger.info(f"Unexpected controller name: {file_prefix}")
                            unexpected_controller_names.append(file_prefix)
                        discovered_controller_names.append(file_prefix)
                        
                        # if file_prefix is in controller_names, add it to the matched_controller_files list
                        if file_prefix in controller_names:
                            matched_controller_files.append(file_prefix)

                        controller_file_path = Path(os.path.join(root, file))
                        controller_files.append(controller_file_path)
                        logger.info(f"Found controller file: {controller_file_path}")
    
        logger.info(f"Found {len(controller_files)} controller files in total")

        # For each name in controller_names, check to see if it exists in matched_controller_files
        all_expected_controllers_found = True
        extra_controllers_found = False
        for name in controller_names:
            if not name in discovered_controller_names:
                logger.info(f"Controller {name}, listed in this project's registry, was not found in filesystem.")
                all_controllers_found = False
        for name in discovered_controller_names:
            if not name in controller_names:
                logger.info(f"Controller {name} was found in filesystem, but was not listed in this project's registry.")
                extra_controllers_found = True

        logger.info(f"Expected to find {len(controller_names)} controller files: {controller_names}. Found: {len(matched_controller_files)}: {matched_controller_files}")
        if (all_expected_controllers_found):
            logger.info("All expected controllers were found in the filesystem.")
        if (extra_controllers_found):
            logger.info("Extra controllers beyond those listed in the registry were found in the filesystem.")
        return controller_files

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
        # Reset visited files for this analysis
        self.visited_files = set()
        # Include the original view file
        dependencies.add(view_file)
        
        

        # add BlueJS component and its dependencies as a dependency
        path_parts = list(view_file.parts)
        adjacent_dir_idx = path_parts.index('views')
        path_parts[adjacent_dir_idx] = 'components'
        component_file = Path(os.sep.join(path_parts))
        if os.path.exists(component_file):
            dependencies.add(Path(component_file))
            logger.info(f"Added component {component_file} to dependencies")
            self._analyze_file_dependencies_recursive(component_file, repo_path, dependencies)

        # Start recursive dependency analysis for the view script
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
                    
                    # Resolve relative imports
                    full_path = self._resolve_dependency_path(imported_path, file_path, repo_path)
                    # logger.info(f"Found dependency: {full_path}")
                    if full_path and Path(full_path).exists():
                        # logger.info(f"Added dependency: {full_path}")
                        dependencies.add(full_path)
                        
                        # Recursively analyze this dependency
                        self._analyze_file_dependencies_recursive(full_path, repo_path, dependencies)
        
        except Exception as e:
            logger.error(f"Error analyzing dependencies for {file_path}: {str(e)}")

    def _convert_to_react(self, view_file: Path, dependent_files_content: Dict[str, str], view_component_dir: Path) -> Dict[str, Any]:
        """
        Use LLM to convert a view file and its dependencies to React components with structured output.
        
        Args:
            view_file: Path to the view file
            dependent_files_content: Dictionary mapping file paths to their content
            view_component_dir: Path where the main component should be placed
            
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
            category: str = Field(description="Category of file: 'component', 'util', 'type', or 'setting'")
            reuse_existing: bool = Field(description="Whether to reuse an existing file instead of creating a new one", default=False)
        
        class ReactConversionFiles(BaseModel):
            files: List[File] = Field(description="List of React component files")
        
        # Configure the LLM for structured output
        structured_llm = self.llm.with_structured_output(ReactConversionFiles)
        
        # Prepare the context for previously converted files
        previously_converted_context = self._get_previously_converted_context()
        
        # Prepare the context for the LLM
        system_prompt = f"""You are an expert at converting legacy web applications to modern React TypeScript applications.
Your task is to convert view files and their dependencies to React components. CONVERT EVERYTHING! DON'T LEAVE GAPS OR PLACEHOLDERS!
Your job is to complete the conversion. Developers will not finish your job for you.

DO NOT DO THINGS LIKE THIS (You should actually write the complete code):
  return (
    <div>
      {{/* Render template and partials */}}
      {{/* Example: <StepOne {{...viewContext}} /> */}}
    </div>
  );

ACTUALLY CONVERT THE TEMPLATES TO REACT COMPONENTS!

Guidelines:
- Convert "context" variables to props
- Convert "model" data to state
- The react components should take individual props, not just a single 'model' object. 
- Convert templates to React JSX components
- For unrecognized web components, import them from '@mds/react' with PascalCase names
- Props for MDS React components should be camelCase
- Return a structured collection of files that together implement the view in React
- Use the MAIN VIEW FILE as the entrypoint
- if templates are large, you may split them into their own granular files.
- For DEPENDENCIES that include "components" in their file path, treat them as utilties and event handlers for the MAIN VIEW FILE. "Components" in the legacy code are NOT the same as React components.
- remove any jQuery or DOM manipulation and use React best practices instead.
- if a template uses the sanitizeHtml function, you can import it like this: `import sanitizeHtml from 'sanitize-html';`

IMPORTANT: REUSE EXISTING FILES
This project already has some previously converted React files. You should:
1. Review the previously converted files listed below
2. REUSE these files when possible instead of creating duplicate implementations
3. Use relative imports to reference these existing files
4. Only create new utility, settings, or type files if you need functionality not available in existing files
5. For each file you create, specify if it should be a new file or reuse an existing one
6. Mark files for reuse by setting reuse_existing=true and provide the same filename as an existing file

{previously_converted_context}

Organize files in the following folder structure:
- components/: Place all React components here (the entrypoint should be directly in this folder)
- utils/: Place utility functions, helpers, and services here
- types/: Place TypeScript interfaces, types, and enums here
- settings/: Place configuration, constants, and default values here

For imports:
- Convert octagon imports like require('@octagon/<package_name>/*') to ES6 import statements (Shouldn't need to import from dist folders now since the new file will use ES6 imports)
- retain external library imports like 'common/lib'. Keep the imports in place, but add a TODO for the dev to revisit those.
- Whenever you see the 'blue/root' library being used, this is just the window object. So instead of using the root object, use window.
- if you do use the window object, make sure it is compatible with server side rendering by checking if it is defined before using it.
- Use relative imports for your own files
- For imports of TypeScript types, specify 'type' explicitly to comply with 'verbatimModuleSyntax'.  For example: "import type {{ ViewContextType }} from '../types/viewContext';"

Example of MDS conversion:
<mds-text-input 
    name="totalMonthlyPayments"
    id="totalMonthlyPayments"
    fieldName="totalMonthlyPayments"
    on-blur="validateAndFormat"
    label={{{{~/totalMonthlyPaymentsLabel}}}}
    placeholder={{{{~/amountPlaceholder}}}}
    on-change="formFieldTracking"
    validate="['required', 'noSpecialCharacters', 'currency', 'positiveNumber', 'noDecimals', 'lessThanOrEqualToMaxNumber', 'greaterThanOrEqualToMinNumber']"
    validate-max="{{{{.maximumInputAmount}}}}"
    validate-min="{{{{.minimumInputAmount}}}}"
    error-key="affordabilityCalculatorQuestionErrorHeader"
    name="totalMonthlyPayments"
    microcopy="{{{{~/totalMonthlyPaymentsAdvisory}}}}"
    value="{{{{sanitize.sanitizeHtml(~/totalMonthlyPayments)}}}}"
>
</mds-text-input>

Becomes:
import {{ MdsTextInput }} from '@mds/react';
import sanitizeHtml from 'sanitize-html';

<MdsTextInput
    name="totalMonthlyPayments"
    id="totalMonthlyPayments"
    fieldName="totalMonthlyPayments"
    onBlur={{validateAndFormat}}
    label={{totalMonthlyPaymentsLabel}}
    placeholder={{amountPlaceholder}}
    onChange={{formFieldTracking}}
    validate={{['required', 'noSpecialCharacters', 'currency', 'positiveNumber', 'noDecimals', 'lessThanOrEqualToMaxNumber', 'greaterThanOrEqualToMinNumber']}}
    validateMax={{maximumInputAmount}}
    validateMin={{minimumInputAmount}}
    errorKey="affordabilityCalculatorQuestionErrorHeader"
    microcopy={{totalMonthlyPaymentsAdvisory}}
    value={{sanitizeHtml(totalMonthlyPayments)}}
/>

IMPORTANT: UPDATING EXISTING FILES
When you encounter functionality that should be added to an existing file:
1. Set reuse_existing=true for that file
2. Include the complete updated file content, not just the new additions
3. Your updated content should intelligently merge with the existing file
4. For utils, types, and shared components, always check if they already exist before creating new ones
5. When adding new functions to existing files, maintain the same style and patterns

For example, if you need to add a formatDate function and there's already a utils/dateUtils.ts file,
add your function to that file rather than creating a new one.
"""
        
        # Find the relative view path
        try:
            # Get a sample path to extract the repo base path
            sample_path = next(iter(dependent_files_content.keys()))
            base_path = os.path.commonpath(list(dependent_files_content.keys()))
            relative_view_path = os.path.relpath(view_file, os.path.join(os.path.dirname(sample_path), base_path))
        except (StopIteration, ValueError):
            # Fallback if we can't determine the relative path
            relative_view_path = os.path.basename(view_file)
        
        user_prompt = f"""Here are the files that need to be converted to React. You MUST convert everything. Do NOT leave gaps or placeholders.

MAIN VIEW FILE:
File: {os.path.basename(view_file)}
```
{dependent_files_content.get(str(view_file), open(view_file, 'r', encoding='utf-8', errors='ignore').read())}
```

DEPENDENCIES:
"""
        
        # Add dependent file contents to the prompt, ensuring we don't duplicate the main view file
        view_file_str = str(view_file)
        for file_path, content in dependent_files_content.items():
            # Skip the main view file as we've already included it
            if file_path != view_file_str and file_path != os.path.basename(view_file):
                user_prompt += f"\nFile: {file_path}\n```\n{content}\n```"
        
        # Call LLM to convert to React with structured output
        try:
            from langchain_core.messages import SystemMessage, HumanMessage

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = None
            
            with get_openai_callback() as cb:
                # Approximate the size of the messages in tokens by counting every 4 characters
                messages_size = sum(len(message.content) for message in messages)
                messages_size = int(messages_size / 4)

                logger.info("Calling LLM (messages size ~" + str(messages_size) + " tokens)")
                response = structured_llm.invoke(messages)
                logger.info(f"Prompt Tokens: {cb.prompt_tokens}")
                logger.info(f"Completion Tokens: {cb.completion_tokens}")
            
            # Convert the structured output to our standard format and update the conversion registry
            result = {
                "components": {}
            }
            
            for file in response.files:
                result["components"][file.filename] = {
                    "content": file.content,
                    "category": file.category,
                    "reuse_existing": file.reuse_existing
                }
                
                # If not reusing, add to our conversion registry
                if not file.reuse_existing:
                    category_key = file.category + "s"  # e.g., "component" -> "components"
                    if category_key in self.converted_files:
                        # Add original files that lead to this conversion as source
                        self.converted_files[category_key][file.filename] = {
                            "content": file.content,
                            "source_files": [str(view_file)] + list(dependent_files_content.keys())
                        }
                
            return result
            
        except Exception as e:
            logger.error(f"Error converting {view_file} to React: {str(e)}")
            return {"error": str(e)}

    def save_react_components(self, components: Dict[str, Dict[str, Any]], source_view_file: Path) -> Dict[str, Any]:
        """
        Save the generated React components to the appropriate directories or update existing files.
        
        Args:
            components: Dictionary of component filenames to their info (content, category, reuse_existing)
            source_view_file: The original view file that was converted
            
        Returns:
            Dictionary with information about saved files
        """
        if not self.project_output_dir:
            raise ValueError("Project output directory not set")
        
        # Save each component
        saved_files = []
        reused_files = []
        updated_files = []
        
        for filename, component_info in components.items():
            if filename == "_explanation" or not component_info.get("content"):
                continue
            
            # Determine the full path based on file category
            category = component_info.get("category", "component")
            
            if category == "component":
                # Components go directly in the components folder or in controller-specific subfolders
                file_path = self.project_output_dir / filename
            else:
                # Utils, types, and settings go in their respective folders
                category_folder = f"{category}s"  # e.g., "util" -> "utils"
                if category_folder not in ["utils", "types", "settings"]:
                    category_folder = "utils"  # Default to utils if unknown
                    
                file_path = self.project_output_dir / category_folder / os.path.basename(filename)
            
            # Create any necessary subdirectories
            os.makedirs(file_path.parent, exist_ok=True)
            
            # Check if file exists and needs updating
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_content = f.read()
                
                if component_info.get("reuse_existing", False):
                    # If marked for reuse but content differs, we should update the file
                    if existing_content.strip() != component_info["content"].strip():
                        merged_content = self._merge_file_contents(existing_content, component_info["content"], filename)
                        
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(merged_content)
                        
                        updated_files.append(str(file_path))
                        logger.info(f"Updated existing file: {file_path}")
                    else:
                        reused_files.append(str(file_path))
                        logger.info(f"Reusing existing file without changes: {file_path}")
                else:
                    # Not marked for reuse, but file exists - we'll update it with new content
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(component_info["content"])
                    updated_files.append(str(file_path))
                    logger.info(f"Replaced existing file: {file_path}")
            else:
                # File doesn't exist, create it
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(component_info["content"])
                
                saved_files.append(str(file_path))
                logger.info(f"Saved {category} to {file_path}")
            
            # Add to our conversion registry if not already there or update existing entry
            category_key = f"{category}s"
            if category_key in self.converted_files:
                # If file already exists in registry, update the content
                if filename in self.converted_files[category_key]:
                    self.converted_files[category_key][filename]["content"] = component_info["content"]
                    # Add this source file to the list if not already there
                    if str(source_view_file) not in self.converted_files[category_key][filename]["source_files"]:
                        self.converted_files[category_key][filename]["source_files"].append(str(source_view_file))
                else:
                    # Add new entry to registry
                    self.converted_files[category_key][filename] = {
                        "content": component_info["content"],
                        "source_files": [str(source_view_file)]
                    }
        
        return {
            "output_dir": str(self.project_output_dir),
            "saved_files": saved_files,
            "reused_files": reused_files,
            "updated_files": updated_files,
            "file_count": len(saved_files) + len(updated_files)
        }

    def convert_views_to_react(self, repo_url: str) -> Dict[str, Any]:
        """
        Find all view files and convert them to React components.
        
        Args:
            repo_url: URL of the git repository to clone and analyze
            
        Returns:
            Dictionary containing conversion results
        """
        logger.info(f"Starting view-to-React conversion for {repo_url}")
        self.start_time = time.time()

        # Ensure the repository storage directory exists
        os.makedirs(REPO_STORAGE_DIR, exist_ok=True)
        
        # Clone the repository to our dedicated storage location
        repo_path = self._clone_repo(repo_url, REPO_STORAGE_DIR)

        # Load any registry files (registry.js, manifest.js) and map controllers to components.
        controller_to_component_map = self._build_controller_to_component_map(repo_path)

        # From controller_to_component_map, create a reciprocal component_to_controller map.
        component_to_controller_map = self._build_component_to_controller_map(controller_to_component_map)
        
        # Find all view files
        view_files = self.find_all_view_files(repo_path)

        # Parse configuration files for path aliases
        self._parse_config_files(repo_path)
        logger.info(f"Parsed configuration files for path aliases. Found {len(self.alias_map)} aliases.")
        
        # Create output directory for React components - ONE directory for the entire project
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        subdir_name = repo_url.split("/")[-1].removesuffix(".git")
        output_base_dir = Path(f"converted_views_{timestamp}")
        self.project_output_dir = output_base_dir.joinpath(subdir_name)
        os.makedirs(self.project_output_dir, exist_ok=True)

        create_octagon_result = subprocess.run(
            [
                "npx", 
                "--yes",
                "@octagon/create-octagon-app@latest",
                "react-app",
                "--contentManagement=none",
                "--deploymentTarget=none",
                "--formManagement=react-hook-form",
                "--microFrontendRender=none",
                "--microFrontends=single-spa",
                f"--name={output_base_dir}",
                "--projectType=standalone",
                "--reporting=none",
                "--services=react-query",
                "--styling=sass",
                "--ui=mds",
                "--uiTesting=none",
                "--uiVariation=none",
                "--utilities=none",
                "--video=none",
              ],
            check=True,
            capture_output=True,
        )
        if create_octagon_result.returncode != 0:
            logger.error(f"Error creating Octagon shell: {create_octagon_result.stderr.decode('utf-8')}")
            return {"error": create_octagon_result.stderr.decode('utf-8')}
        
        logger.info("Adding common dependencies ...")
        # add common dependencies
        add_cxo_common_result = subprocess.run(
            [
                "npm",
                "add",
                "@seur/cxo-ui-common-utilities",
                "@seur/cxo-common-assets",
                "@mds/react",
                "d3"
            ],
            cwd=Path.cwd().joinpath(output_base_dir),
            check=True,
            capture_output=True,
        )

        if add_cxo_common_result.returncode != 0:
            logger.error(f"Error adding cxo common: {add_cxo_common_result.stderr.decode('utf-8')}")
            return {"error": add_cxo_common_result.stderr.decode('utf-8')}  
        
        if add_cxo_common_result.stdout:
            logger.info(f"Added common dependencies: {add_cxo_common_result.stdout.decode('utf-8')}")

        # Create standard directories in the project output directory
        for dir_name in ["components", "utils", "types", "settings"]:
            os.makedirs(self.project_output_dir / dir_name, exist_ok=True)

        # Process each view file
        conversion_results = {}
        generated_file_count = 0
        current_view_file_counter = 0
        
        # Sort view files by how frequently they're depended on to convert shared utilities first
        view_files_with_scores = self._score_view_files_by_dependencies(view_files, repo_path)
        
        for view_file, _ in view_files_with_scores:
            try:
                current_view_file_counter += 1
                logger.info(f"Processing view file {current_view_file_counter}/{len(view_files)}: {view_file}")
                view_name = os.path.splitext(os.path.basename(view_file))[0]
                
                # Find parent controller
                parent_controller = "no_controller"
                if component_to_controller_map is not None:
                    parent_controller = component_to_controller_map.get(view_name, "no_controller")
                
                # Set target directory for this view's main component
                view_component_dir = self.project_output_dir / "components" / parent_controller / view_name
                os.makedirs(view_component_dir, exist_ok=True)
                
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
                
                # Convert to React using LLM with context about already converted files
                react_components = self._convert_to_react(view_file, dependent_files_content, view_component_dir)
                
                # Save the React components
                if "components" in react_components:
                    save_result = self.save_react_components(react_components["components"], view_file)
                    generated_file_count += save_result["file_count"]
                    react_components["save_result"] = save_result
                    
                    # Log information about file updates
                    if save_result.get("updated_files"):
                        logger.info(f"Updated {len(save_result['updated_files'])} existing files")
                
                # Store results
                relative_view_path = os.path.relpath(view_file, repo_path)
                conversion_results[relative_view_path] = {
                    "original_view": str(view_file),
                    "dependencies": [str(dep) for dep in dependencies],
                    "react_components": react_components,
                    "output_dir": str(view_component_dir)
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
            "generated_file_count": generated_file_count,
            "analysis_time_seconds": elapsed_time,
            "controller_to_component_map": controller_to_component_map,
            "output_base_dir": str(output_base_dir),
            "reused_file_count": self._count_reused_files(),
            "updated_file_count": sum(
                len(result.get("react_components", {}).get("save_result", {}).get("updated_files", []))
                for result in conversion_results.values()
                if isinstance(result, dict) and "react_components" in result
            )
        }

    def _score_view_files_by_dependencies(self, view_files, repo_path):
        """
        Score view files based on how often they're depended on, to prioritize shared utilities.
        
        Args:
            view_files: List of view file paths
            repo_path: Path to repository
            
        Returns:
            List of (view_file, score) tuples sorted by score (highest first)
        """
        dependency_counts = {}
        file_dependency_map = {}
        
        # First, collect all dependencies for each view file
        for view_file in view_files:
            deps = self.analyze_view_dependencies(view_file, repo_path)
            file_dependency_map[view_file] = deps
            
            # Count how many times each file is depended on
            for dep in deps:
                if dep not in dependency_counts:
                    dependency_counts[dep] = 0
                dependency_counts[dep] += 1
        
        # Now score each view file based on how many times its dependencies are used elsewhere
        scored_files = []
        for view_file in view_files:
            # Base score for the file itself
            score = dependency_counts.get(view_file, 0)
            
            # Add scores for its dependencies
            for dep in file_dependency_map.get(view_file, []):
                score += dependency_counts.get(dep, 0)
            
            scored_files.append((view_file, score))
        
        # Sort by score, highest first
        return sorted(scored_files, key=lambda x: x[1], reverse=True)

    def _count_reused_files(self):
        """Count how many files were reused instead of regenerated."""
        reuse_count = 0
        for category in self.converted_files.values():
            reuse_count += len(category)
        return reuse_count

    def _get_previously_converted_context(self) -> str:
        """
        Create a context string describing previously converted files.
        
        Returns:
            String describing previously converted files to include in the prompt
        """
        context_parts = ["PREVIOUSLY CONVERTED FILES:"]
        
        # List converted utilities
        if self.converted_files["utils"]:
            context_parts.append("\nUTILITY FILES:")
            for filename, file_info in self.converted_files["utils"].items():
                # Get a short preview of the file
                content = file_info["content"]
                preview = content[:500] + "..." if len(content) > 500 else content
                context_parts.append(f"\nFile: {filename}\n```typescript\n{preview}\n```")
        
        # List converted types
        if self.converted_files["types"]:
            context_parts.append("\nTYPE FILES:")
            for filename, file_info in self.converted_files["types"].items():
                content = file_info["content"]
                preview = content[:500] + "..." if len(content) > 500 else content
                context_parts.append(f"\nFile: {filename}\n```typescript\n{preview}\n```")
        
        # List converted settings
        if self.converted_files["settings"]:
            context_parts.append("\nSETTINGS FILES:")
            for filename, file_info in self.converted_files["settings"].items():
                content = file_info["content"]
                preview = content[:500] + "..." if len(content) > 500 else content
                context_parts.append(f"\nFile: {filename}\n```typescript\n{preview}\n```")
        
        # List a few key components that might be reusable
        if self.converted_files["components"]:
            context_parts.append("\nKEY COMPONENT FILES:")
            # Only include a few component files to avoid prompt size issues
            component_count = 0
            for filename, file_info in self.converted_files["components"].items():
                if component_count >= 5:  # Limit to 5 components
                    break
                
                # Only include smaller components that are likely to be reusable
                content = file_info["content"]
                if "type Props" in content or "interface Props" in content:
                    preview = content[:500] + "..." if len(content) > 500 else content
                    context_parts.append(f"\nFile: {filename}\n```typescript\n{preview}\n```")
                    component_count += 1
        
        if len(context_parts) == 1:
            return "No previously converted files yet."
        
        return "\n".join(context_parts)

    def _merge_file_contents(self, existing_content: str, new_content: str, filename: str) -> str:
        """
        Merge existing file content with new content, preserving manual edits.
        
        Args:
            existing_content: Content of existing file
            new_content: Generated new content 
            filename: Name of the file being merged
            
        Returns:
            Merged content
        """
        # For TypeScript/JavaScript files, use more intelligent merging
        if filename.endswith(('.ts', '.tsx', '.js', '.jsx')):
            try:
                # Use the LLM to help with merging
                merge_prompt = f"""
                I need to merge an existing file with newly generated content.
                
                EXISTING FILE:
                ```
                {existing_content}
                ```
                
                NEW CONTENT TO MERGE:
                ```
                {new_content}
                ```
                
                Please merge these files intelligently:
                1. Keep all imports from both files (deduplicated)
                2. Merge interface/type definitions, adding new fields from the new content
                3. Keep all existing functions, but update any that have the same name in the new content
                4. Add any completely new functions from the new content
                
                Return ONLY the merged content without any explanations.
                """
                merged_content = self.llm.invoke([HumanMessage(content=merge_prompt)]).content
                
                # Clean up any markdown formatting
                merged_content = re.sub(r'^```(?:jsx|javascript|typescript|tsx|react)?\s*', '', merged_content, flags=re.MULTILINE)
                merged_content = re.sub(r'\s*```$', '', merged_content, flags=re.MULTILINE)
                
                logger.info(f"Intelligently merged file: {filename}")
                return merged_content
            except Exception as e:
                logger.error(f"Error merging file {filename}: {e}")
                # Fall back to new content if merging fails
                return new_content
        else:
            # For other file types, use the new content
            return new_content


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

    logger.info(f"Converted {result['view_count']} view(s) to {result['generated_file_count']} React components in {result['analysis_time_seconds']:.2f} seconds.")
    logger.info(f"Wrote results to: {result['output_base_dir']}/")