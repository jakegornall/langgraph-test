"""Development server for testing React components."""

import os
import tempfile
import subprocess
import shutil
import json
import traceback
from pathlib import Path
from typing import Dict, Any, List

from react_agent.models import FileItem

VITE_CONFIG = """
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: parseInt(process.env.PORT || '3000'),
  }
})
"""

PACKAGE_JSON = {
    "name": "react-component-preview",
    "version": "0.0.0",
    "type": "module",
    "scripts": {
        "dev": "vite",
        "build": "tsc && vite build",
        "preview": "vite preview"
    },
    "dependencies": {
        "react": "^18.2.0",
        "react-dom": "^18.2.0",
        "@mui/material": "^5.15.11",
        "@mui/icons-material": "^5.15.11",
        "@emotion/react": "^11.11.3",
        "@emotion/styled": "^11.11.0",
        "@fontsource/roboto": "^5.0.8"
    },
    "devDependencies": {
        "@types/react": "^18.2.43",
        "@types/react-dom": "^18.2.17",
        "@vitejs/plugin-react": "^4.2.1",
        "typescript": "^5.2.2",
        "vite": "^5.0.8"
    }
}

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Component Preview</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
"""

MAIN_TSX = """
import React from 'react'
import { createRoot } from 'react-dom/client'
import '@fontsource/roboto/300.css';
import '@fontsource/roboto/400.css';
import '@fontsource/roboto/500.css';
import '@fontsource/roboto/700.css';
import { ThemeProvider, createTheme } from '@mui/material';
import CssBaseline from '@mui/material/CssBaseline';
import Component from './component'

const theme = createTheme();

const root = createRoot(document.getElementById('root')!);
root.render(
  <React.StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Component />
    </ThemeProvider>
  </React.StrictMode>
);
"""


class DevServer:
    """Manages a development server for testing React components."""
    
    def __init__(self, port: int = 3002):
        self.port = port
        self.temp_dir = None
        self.process = None

    def _check_npm_installed(self) -> bool:
        """Check if npm is available in the system."""
        try:
            cmd = 'where' if os.name == 'nt' else 'which'
            subprocess.run([cmd, 'npm'], check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def _merge_dependencies(self, package_json: dict, dependencies: List[str]) -> dict:
        """Merge additional dependencies into package.json."""
        merged = package_json.copy()
        for dep in dependencies:
            parts = dep.strip().split('@', 1)
            name = parts[0].strip()
            version = parts[1].strip() if len(parts) > 1 else 'latest'
            merged['dependencies'][name] = version
        return merged

    def setup(self, files: Dict[str, List[FileItem]], dependencies: List[str] = None) -> None:
        """Set up the development environment with component files and a single project-level list of dependencies."""
        if not self._check_npm_installed():
            raise RuntimeError("npm is not installed or not found in PATH.")

        try:
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp(prefix='react_dev_')
            src_dir = Path(self.temp_dir) / "src"
            src_dir.mkdir(parents=True)

            print(f"Created temporary directory: {self.temp_dir}")

            # Merge additional dependencies if provided
            merged_package_json = PACKAGE_JSON
            if dependencies:
                merged_package_json = self._merge_dependencies(merged_package_json, dependencies)

            # Write configuration files
            with open(Path(self.temp_dir) / "package.json", "w", encoding="utf-8") as f:
                json.dump(merged_package_json, f, indent=2)

            with open(Path(self.temp_dir) / "vite.config.ts", "w", encoding="utf-8") as f:
                f.write(VITE_CONFIG)

            with open(Path(self.temp_dir) / "index.html", "w", encoding="utf-8") as f:
                f.write(INDEX_HTML)

            with open(src_dir / "main.tsx", "w", encoding="utf-8") as f:
                f.write(MAIN_TSX)

            # Write component files
            for file in files["files"]:
                file_path = src_dir / file.filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as handle:
                    handle.write(file.content)

            # Install dependencies with better error handling
            use_shell = os.name == 'nt'
            try:
                result = subprocess.run(
                    ["npm", "install"],
                    cwd=str(self.temp_dir),
                    check=True,
                    capture_output=True,
                    text=True,
                    shell=use_shell
                )
                print(f"npm install stdout: {result.stdout}")
                print(f"npm install stderr: {result.stderr}")
            except subprocess.CalledProcessError as e:
                print(f"npm install failed with exit code {e.returncode}")
                print(f"stdout: {e.stdout}")
                print(f"stderr: {e.stderr}")
                raise RuntimeError(f"npm install failed:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")

        except Exception as e:
            if self.temp_dir and Path(self.temp_dir).exists():
                shutil.rmtree(self.temp_dir)
            print(f"Setup failed: {str(e)}")
            print(f"Stack trace: {''.join(traceback.format_tb(e.__traceback__))}")
            raise RuntimeError(
                f"Failed to set up development environment:\n{str(e)}\n{''.join(traceback.format_tb(e.__traceback__))}"
            )

    def start(self) -> Dict[str, Any]:
        """Start the development server."""
        try:
            use_shell = os.name == 'nt'
            self.process = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=self.temp_dir,
                env={**os.environ, "PORT": str(self.port)},
                shell=use_shell,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            import time
            time.sleep(2)  # Give the server time to launch

            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                return {
                    "success": False,
                    "errors": [
                        f"Server failed to start:\nSTDOUT:\n{stdout.decode()}\nSTDERR:\n{stderr.decode()}"
                    ]
                }

            return {
                "success": True,
                "dev_server_url": f"http://localhost:{self.port}"
            }

        except Exception as e:
            return {
                "success": False,
                "errors": [f"{str(e)}\n{''.join(traceback.format_tb(e.__traceback__))}"]
            }

    def stop(self) -> None:
        """Stop the development server and clean up."""
        if self.process:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(self.process.pid)])
            else:
                self.process.terminate()
            self.process.wait()

        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir) 