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

NPMRC = """registry=https://artifacts.jpmchase.net/artifactory/api/npm/npm/"""

TSCONFIG = {
    "compilerOptions": {
        "target": "ES2022",
        "module": "ESNext",
        "jsx": "react-jsx",
        "jsxImportSource": "react",
        "sourceMap": True,
        "outDir": "dist",
        "rootDir": "src",
        "strict": True,
        "moduleResolution": "node",
        "esModuleInterop": True,
        "skipLibCheck": True,
        "forceConsistentCasingInFileNames": True,
        "resolveJsonModule": True,
        "isolatedModules": True,
        "noEmit": True,
        "lib": ["ES2022", "DOM"],
        "paths": {
            "@/*": ["./src/*"]
        }
    },
    "include": ["src/**/*.ts", "src/**/*.tsx", "src/**/*.json"],
    "exclude": ["node_modules", "**/node_modules/*"]
}

PACKAGE_JSON = {
    "name": "react-component-preview",
    "version": "0.0.0",
    "type": "module",
    "scripts": {
        "dev": "vite",
        "build": "tsc --noEmit &&vite build",
        "preview": "vite preview"
    },
    "dependencies": {
        "react": "^18.2.0",
        "react-dom": "^18.2.0",
        "@mds/web-ui-theme": "^3.2.0",
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
import { Theme } from '@mds/web-ui-theme'; // <-- import the component
import '@mds/web-ui-theme/cmb';
import Component from './component'

const root = createRoot(document.getElementById('root')!);
root.render(
  <React.StrictMode>
    <Theme>
      <Component />
    </Theme>
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

    def setup(self, files: Dict[str, List[FileItem]], dependencies: List[str] = None) -> dict:
        """Set up the development environment with component files and a single project-level list of dependencies."""
        if not self._check_npm_installed():
            return {
                "success": False,
                "message": "npm is not installed or not found in PATH."
            }

        result = {
            "success": True,
            "message": "",
        }

        try:
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp(prefix='react_dev_')
            src_dir = Path(self.temp_dir) / "src"
            src_dir.mkdir(parents=True)

            print(f"Created temporary directory: {self.temp_dir}")

            # Write configuration files
            with open(Path(self.temp_dir) / "package.json", "w", encoding="utf-8") as f:
                json.dump(PACKAGE_JSON, f, indent=2)

            with open(Path(self.temp_dir) / "vite.config.ts", "w", encoding="utf-8") as f:
                f.write(VITE_CONFIG)
            
            with open(Path(self.temp_dir) / "tsconfig.json", "w", encoding="utf-8") as f:
                json.dump(TSCONFIG, f, indent=2)

            with open(Path(self.temp_dir) / ".npmrc", "w", encoding="utf-8") as f:
                f.write(NPMRC)

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
                print(f"npm install in {self.temp_dir}")
                result_install = subprocess.run(
                    ["npm", "install"],
                    cwd=str(self.temp_dir),
                    check=True,
                    capture_output=True,
                    text=True,
                    shell=use_shell
                ).stdout.strip()
                result["message"] += result_install

                # Merge additional dependencies if provided
                if dependencies:
                    for dep in dependencies:
                        print(f"npm install {dep}")
                        try:
                            result_dep = subprocess.run(['npm', 'install', dep], check=True, cwd=self.temp_dir, capture_output=True, text=True).stdout.strip()
                            print(result_dep)
                            result["message"] += result_dep
                        except subprocess.CalledProcessError as e:
                            result["success"] = False
                            result["message"] += f"\nnpm install failed: {e.stdout.strip()}\n{e.stderr.strip()}"
                            print(f"npm install failed: {e.stdout.strip()}\n{e.stderr.strip()}")
                    result['success'] = True

            except subprocess.CalledProcessError as e:
                result["success"] = False
                result["message"] += f"\nnpm install failed: {e.stdout.strip()}\n{e.stderr.strip()}"

        except Exception as e:
            result["success"] = False
            result["message"] += str(e)

        return result

    def build(self) -> str:
        """Build the development environment."""
        try:
            result_build = subprocess.run(["npm", "run", "build"], cwd=self.temp_dir, check=True, capture_output=True, text=True).stdout.strip()
            print(f"Build result: {result_build}")
            return {
                "success": True,
                "message": result_build
            }
        except subprocess.CalledProcessError as e:
            print(f"Build failed with following error:\nSTDOUT:\n{e.stdout.strip()}\nSTDERR:\n{e.stderr.strip()}")
            return {
                "success": False,
                "message":f"Build failed with following error:\nSTDOUT:\n{e.stdout.strip()}\nSTDERR:\n{e.stderr.strip()}"
            }

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
            time.sleep(2)

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


if __name__ == "__main__":
    INDEX_TSX = """import React from 'react';
import { Button } from '@mds/web-ui-button';

const App = () => {
  return (
    <div>
        <Button text="Hello World" />
    </div>
  )
}

export default App;
"""
    file_item = FileItem()
    file_item.filename = "index.tsx"
    file_item.content = INDEX_TSX
    file_item.file_type = "tsx"
    file_item.entrypoint = True

    dev_server = DevServer()
    dev_server.setup({ "files": [
        file_item,
    ] }, dependencies=["@mds/web-ui-theme", "@mds/web-ui-button"])
    result = dev_server.start()
    print(result)