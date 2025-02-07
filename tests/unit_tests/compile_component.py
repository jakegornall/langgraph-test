import pytest
from react_agent.tools import compile_component
from react_agent.utils import FileContent
from unittest.mock import Mock, patch

@pytest.fixture
def valid_component():
    return [
        FileContent(
            filename="Button.tsx",
            content="""
                import React from 'react';
                export const Component = () => {
                    return <button>Click me</button>;
                };
            """,
            file_type="tsx",
            entrypoint=True
        )
    ]

@pytest.fixture
def invalid_component():
    return [
        FileContent(
            filename="Button.tsx",
            content="""
                import React from 'react';
                export const Component = () => {
                    return <button>Click me</button>  // Missing semicolon
                };
            """,
            file_type="tsx",
            entrypoint=True
        )
    ]

def test_compile_component_success(valid_component):
    with patch('react_agent.tools.DevServer') as MockDevServer:
        # Configure mock
        mock_server = Mock()
        MockDevServer.return_value = mock_server
        mock_server.start.return_value = {
            "success": True,
            "dev_server_url": "http://localhost:3000"
        }

        # Call function
        result = compile_component(valid_component, config={})

        # Verify results
        assert result["success"] is True
        assert result["dev_server_url"] == "http://localhost:3000"

        # Verify server interactions
        mock_server.setup.assert_called_once()
        mock_server.start.assert_called_once()
        mock_server.stop.assert_not_called()

def test_compile_component_server_error(valid_component):
    with patch('react_agent.tools.DevServer') as MockDevServer:
        # Configure mock
        mock_server = Mock()
        MockDevServer.return_value = mock_server
        mock_server.start.return_value = {
            "success": False,
            "errors": ["Server failed to start"]
        }

        # Call function
        result = compile_component(valid_component, config={})

        # Verify results
        assert "errors" in result
        assert result["errors"] == ["Server failed to start"]

        # Verify server interactions
        mock_server.setup.assert_called_once()
        mock_server.start.assert_called_once()
        mock_server.stop.assert_called_once()

def test_compile_component_setup_exception(valid_component):
    with patch('react_agent.tools.DevServer') as MockDevServer:
        # Configure mock
        mock_server = Mock()
        MockDevServer.return_value = mock_server
        mock_server.setup.side_effect = Exception("Setup failed")

        # Call function
        result = compile_component(valid_component, config={})

        # Verify results
        assert "errors" in result
        assert "Setup failed" in result["errors"][0]

        # Verify server interactions
        mock_server.setup.assert_called_once()
        mock_server.start.assert_not_called()

def test_compile_component_file_renaming(valid_component):
    with patch('react_agent.tools.DevServer') as MockDevServer:
        # Configure mock
        mock_server = Mock()
        MockDevServer.return_value = mock_server
        mock_server.start.return_value = {"success": True, "dev_server_url": "http://localhost:3000"}

        # Call function
        compile_component(valid_component, config={})

        # Verify file renaming
        setup_call_args = mock_server.setup.call_args[0][0]
        assert any(f.filename == "component/index.tsx" for f in setup_call_args["files"]) 