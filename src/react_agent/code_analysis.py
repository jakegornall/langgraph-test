"""
Code analysis module to extract requirements and convert BlueJS applications to React.
"""

import os
import subprocess
import tempfile
import shutil
from typing import Dict, List, Tuple, Optional, Any, Set
import re
from pathlib import Path
import json
import time
import logging
from functools import lru_cache
import hashlib

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from react_agent.ChaseAzureOpenAI import getModel
from langchain.memory import ConversationBufferMemory

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
        
    @lru_cache(maxsize=100)
    def _cached_llm_call(self, prompt_key, **kwargs):
        """Cache LLM calls to avoid repetitive API requests."""
        # This is a simplified approach - in production you might 
        # want a more robust caching mechanism
        cache_key = f"{prompt_key}:{hash(frozenset(kwargs.items()))}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        # Add jitter to avoid rate limits
        time.sleep(0.1)
        
        # Check for timeout
        if self.start_time and time.time() - self.start_time > self.timeout:
            raise TimeoutError("Analysis timed out. Consider analyzing fewer files or increasing timeout.")
        
        return None  # No cached result
    
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
        
        # Find controller file
        controller_file = self._find_controller_file(repo_path, controller_name)
        if not controller_file:
            raise ValueError(f"Could not find controller file for {controller_name}")
            
        logger.info(f"Found controller file: {controller_file}")
        
        # Analyze the controller file and its dependencies
        controller_analysis = self._analyze_file_dependencies(controller_file, repo_path)
        
        # Find action function in controller
        action_code, action_details = self._find_action_function(controller_file, action)
        if not action_code:
            raise ValueError(f"Could not find action '{action}' in controller {controller_file}")
                
        logger.info(f"Found action function: {action}")
        
        # Find templates used by this action and related files
        templates = self._find_templates(repo_path, action_code, controller_file, action_details)
        logger.info(f"Found {len(templates)} templates")
        
        # Convert templates to React components
        react_components = self._convert_templates_to_react(templates)
        
        # Analyze code to generate requirements with dependency context
        requirements = self._generate_requirements_with_context(
            repo_path, 
            controller_file, 
            action_code, 
            file_tree, 
            templates,
            controller_analysis
        )
        
        elapsed_time = time.time() - self.start_time
        logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
        
        return {
            "screen_id": screen_id,
            "file_tree": file_tree,
            "controller_file": str(controller_file),
            "action_function": action,
            "action_code": action_code,
            "action_details": action_details,
            "templates": templates,
            "react_components": react_components,
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
    
    def _build_file_tree(self, root_path: str) -> str:
        """Build a string representation of the directory/file tree."""
        result = []
        
        def traverse(path, prefix=""):
            entries = sorted(os.listdir(path))
            for i, entry in enumerate(entries):
                is_last = i == len(entries) - 1
                entry_path = os.path.join(path, entry)
                
                # Skip .git directory and other hidden files
                if entry.startswith('.'):
                    continue
                
                # Add to result
                connector = "└── " if is_last else "├── "
                result.append(f"{prefix}{connector}{entry}")
                
                # Recursively process directory
                if os.path.isdir(entry_path):
                    next_prefix = prefix + ("    " if is_last else "│   ")
                    traverse(entry_path, next_prefix)
        
        traverse(root_path)
        return "\n".join(result)
    
    def _find_controller_file(self, repo_path: str, controller_name: str) -> Optional[Path]:
        """
        Find the controller file in the repository based on the controller name.
        
        Args:
            repo_path: Path to the repository root
            controller_name: Name of the controller to find
            
        Returns:
            Path to the controller file or None if not found
        """
        logger.info(f"Looking for controller file with name: {controller_name}")
        
        # Common test patterns to exclude
        test_patterns = [
            "/test/", "/tests/", "/spec/", "/__tests__/", "/mocks/", "/__mocks__/",
            "/fixtures/", "/__fixtures__/", "/stubs/", "/__stubs__/",
            ".test.", ".spec.", ".mock.", ".stub.", ".fixture."
        ]
        
        # Common controller name patterns
        controller_patterns = [
            f"{controller_name}.js",
            f"{controller_name}Controller.js",
            f"{controller_name}_controller.js",
            f"{controller_name}-controller.js"
        ]
        
        # Look for the controller file in the repository
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden directories and test directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and 
                      not d in ['test', 'tests', 'spec', '__tests__', 'mocks', '__mocks__',
                               'fixtures', '__fixtures__', 'stubs', '__stubs__']]
            
            for file in files:
                # Skip files that don't match the controller pattern
                if not any(file.lower() == pattern.lower() for pattern in controller_patterns):
                    continue
                    
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, repo_path)
                
                # Skip test files based on path patterns
                if any(test_pattern in relative_path.lower() for test_pattern in test_patterns):
                    logger.debug(f"Skipping test file: {relative_path}")
                    continue
                
                logger.info(f"Found controller file: {file_path}")
                return Path(file_path)
        
        # If not found with exact name, look for files that might contain the controller name
        logger.info(f"Controller file not found with exact name, searching for partial matches...")
        
        # Expanded search for controller-like files
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden directories and test directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and 
                      not d in ['test', 'tests', 'spec', '__tests__', 'mocks', '__mocks__',
                               'fixtures', '__fixtures__', 'stubs', '__stubs__']]
            
            for file in files:
                if file.endswith('.js') and controller_name.lower() in file.lower():
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, repo_path)
                    
                    # Skip test files based on path patterns
                    if any(test_pattern in relative_path.lower() for test_pattern in test_patterns):
                        logger.debug(f"Skipping test file: {relative_path}")
                        continue
                    
                    # Confirm this looks like a controller by checking file contents
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        # Check if file has controller-like patterns
                        if 'define(' in content and ('function(' in content or '=>' in content):
                            logger.info(f"Found potential controller file: {file_path}")
                            return Path(file_path)
        
        logger.warning(f"Could not find controller file for {controller_name}")
        return None
    
    def _find_action_function(self, controller_file: Path, action_name: str) -> Tuple[str, Dict[str, Any]]:
        """Extract the action function and its details from the controller file."""
        with open(controller_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Use AI to extract the action function
        action_extraction_prompt = PromptTemplate.from_template(
            """You are analyzing a BlueJS controller file. Extract the '{action_name}' action function.
            
            Controller file content:
            ```javascript
            {file_content}
            ```
            
            1. Extract the complete code for the '{action_name}' function
            2. Identify any relevant details:
               - Templates it renders
               - API endpoints it calls
               - Data models it interacts with
               - Parameters it uses
            
            Format your response as JSON with the following structure:
            {{
                "function_code": "full function code here",
                "templates": ["list of template names"],
                "api_endpoints": ["list of API endpoints"],
                "data_models": ["list of data models"],
                "parameters": ["list of parameters"]
            }}
            """
        )
        
        message_content = action_extraction_prompt.format(file_content=content)
        response = self.llm.invoke([HumanMessage(content=message_content)])
        
        result = json.loads(response.content)
        
        return result.get("function_code", ""), result
    
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
                    """Convert this BlueJS template to a React component with Material-UI:
                    
                    Template name: {name}
                    Template path: {path}
                    
                    ```html
                    {content}
                    ```
                    
                    Please convert this to a modern React component:
                    1. Use functional components with hooks
                    2. Use Material-UI for styling and components
                    3. Handle any mustache/handlebars expressions properly
                    4. Convert BlueJS specific patterns to React patterns
                    5. Ensure proper prop handling
                    
                    Return just the React component code without any explanations.
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
    
    def _analyze_file_dependencies(self, file_path: Path, repo_path: str, depth: int = 0) -> Dict[str, Any]:
        """
        Recursively analyze a file and its dependencies.
        
        Args:
            file_path: Path to the file to analyze
            repo_path: Path to the repository root
            depth: Current recursion depth to prevent too deep analysis
            
        Returns:
            Dictionary with file analysis information
        """
        if depth > 5:  # Limit recursion depth
            return {"message": "Max recursion depth reached"}
            
        if file_path in self.visited_files:
            return {"message": "Already visited"}
            
        self.visited_files.add(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            return {"error": f"Could not read file {file_path}: {str(e)}"}
            
        # Extract define/require dependencies
        dependencies = self._extract_dependencies(content, file_path)
        self.dependency_graph[str(file_path)] = dependencies
        
        # Use LLM to analyze the file content
        analysis_prompt = PromptTemplate.from_template(
            """Analyze this JavaScript file from a BlueJS application:
            
            File path: {file_path}
            
            ```javascript
            {content}
            ```
            
            Focus on:
            1. What does this module do?
            2. What important state or data does it manage?
            3. What APIs or services does it interact with?
            4. Are there any UI components or templates used?
            5. How might this translate to React patterns?
            
            Be concise but comprehensive.
            """
        )
        
        message_content = analysis_prompt.format(file_path=str(file_path), content=content)
        response = self.llm.invoke([HumanMessage(content=message_content)])
        
        analysis = response.content
        
        # Save analysis to memory
        self.memory.save_context(
            {"input": f"Analyzed file: {file_path}"}, 
            {"output": analysis}
        )
        
        # If there are dependencies, let the LLM prioritize which ones to analyze next
        dependency_analysis = {}
        if dependencies:
            # Resolve all dependency paths
            resolved_dependencies = {}
            for dep in dependencies:
                dep_path = self._resolve_dependency_path(dep, file_path, repo_path)
                if dep_path and os.path.exists(dep_path):
                    # Get a preview of the dependency content
                    try:
                        with open(dep_path, 'r', encoding='utf-8', errors='ignore') as f:
                            preview = f.read(500)  # Just read the first 500 chars for a preview
                        resolved_dependencies[dep] = {
                            "path": dep_path,
                            "preview": preview
                        }
                    except Exception:
                        pass  # Skip if we can't read the file
            
            if resolved_dependencies:
                # Ask LLM to prioritize dependencies
                prioritization_prompt = PromptTemplate.from_template(
                    """You're analyzing dependencies for a JavaScript file in a BlueJS application.
                    
                    Current file being analyzed: {file_path}
                    File analysis: {file_analysis}
                    
                    The file has the following dependencies:
                    {dependencies}
                    
                    Based on the file's purpose and the preview of each dependency, 
                    select up to 3 most important dependencies to analyze next that would help understand:
                    1. Core application logic
                    2. Data flow and state management
                    3. UI components and rendering
                    4. API interactions
                    
                    Return a JSON array of dependency names in priority order, with the most important first.
                    Only include dependencies that would be valuable to analyze deeper. Format: ["dep1", "dep2", "dep3"]
                    """
                )
                
                # Format dependencies for the prompt
                dep_formatted = "\n\n".join([
                    f"Dependency: {dep}\nPath: {info['path']}\nPreview:\n```javascript\n{info['preview']}...\n```" 
                    for dep, info in resolved_dependencies.items()
                ])
                
                json_content = prioritization_prompt.format(
                    file_path=str(file_path),
                    file_analysis=analysis,
                    dependencies=dep_formatted
                )
                
                structured_llm = self.llm.with_structured_output(list[str])
                prioritized_deps = structured_llm.invoke(json_content)
                
                # Analyze prioritized dependencies
                for dep in prioritized_deps:
                    if dep in resolved_dependencies:
                        dep_path = resolved_dependencies[dep]["path"]
                        dependency_analysis[dep] = self._analyze_file_dependencies(
                            Path(dep_path), repo_path, depth + 1
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
                    full_path = os.path.join(alias_path, relative_path + ext)
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
        action_code: str, 
        file_tree: str, 
        templates: List[Dict[str, str]],
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
            
            Main action function:
            ```javascript
            {action_code}
            ```
            
            Dependency Analysis:
            {dependency_analysis}
            
            Previous file analysis:
            {analysis_context}
            
            Template information:
            {template_info}
            
            Path aliases:
            {config_info}
            
            Based solely on the code patterns you observe (not on prior knowledge of BlueJS), create a requirements document with these sections:
            1. Application Overview - What appears to be the purpose of this application based on the code?
            2. Data Flow and State Management - How does the application manage data? What state management would be needed in React?
            3. UI Components - What UI components will need to be created in React based on the templates?
            4. API Integrations - What external services or APIs does the application interact with?
            5. Routing Structure - How does routing appear to work in this application?
            6. React Implementation Approach - How would you recommend implementing this in React?
            
            Focus on concrete patterns you observe in the code without making assumptions about how BlueJS works internally.
            """
        )
        
        template_info = "\n\n".join([
            f"Template: {t['name']}\nPath: {t['path']}\nContent (snippet):\n{t['content'][:300]}..."
            for t in templates
        ]) if templates else "No templates found."
        
        config_info = "\n".join([
            f"- {c['alias']} -> {c['path']}"
            for c in config_files
        ]) if config_files else "No path aliases found."
        
        message_content = requirements_prompt.format(
            file_tree=file_tree,
            controller_file=str(controller_file),
            controller_content=controller_content,
            action_code=action_code,
            dependency_analysis=json.dumps(controller_analysis, indent=2),
            analysis_context=analysis_context,
            template_info=template_info,
            config_info=config_info
        )
        
        requirements = self.llm.invoke([HumanMessage(content=message_content)]).content
        
        return {
            "full_requirements": requirements
        }


def analyze_blueJS_repo(repo_url: str, screen_id: str) -> Dict[str, Any]:
    """
    Analyze a BlueJS repository to extract requirements and convert to React.
    
    Args:
        repo_url: URL of the git repository to clone and analyze
        screen_id: Path-like string in format "<app_name>/<area_name>/<controller_file_name>/<action>"
        
    Returns:
        Dictionary containing analysis results, requirements and converted React components
    """
    analyzer = CodeAnalyzer()
    return analyzer.analyze_repo(repo_url, screen_id)


if __name__ == "__main__":
    # Import required modules for the main function
    from dotenv import load_dotenv
    import os
    import sys
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get repo_url and screen_id from environment variables
    repo_url = os.getenv("REPO_URL")
    screen_id = os.getenv("SCREEN_ID")
    
    # Validate that we have the required environment variables
    if not repo_url or not screen_id:
        print("Error: REPO_URL and SCREEN_ID must be set in .env file")
        sys.exit(1)
    
    # Run the analysis
    result = analyze_blueJS_repo(repo_url, screen_id)
    print(result)

