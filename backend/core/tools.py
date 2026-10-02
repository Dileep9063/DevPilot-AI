from pathlib import Path
import contextvars
import difflib
from langchain_core.tools import tool
import os
import subprocess
import sys

DEFAULT_WORKSPACE = Path("workspace")
_workspace_context = contextvars.ContextVar(
    "devpilot_workspace",
    default=DEFAULT_WORKSPACE,
)

def set_workspace(path):
    """Set the workspace used by tools for the current agent invocation."""
    return _workspace_context.set(Path(path))

def reset_workspace(token):
    """Restore the previous workspace context."""
    _workspace_context.reset(token)

def get_workspace():
    """Return the workspace for the current agent invocation."""
    return _workspace_context.get()

# Backward-compatible name used by tests and existing code.
WORKSPACE = DEFAULT_WORKSPACE
def safe_workspace_path(path: str):
    """
    Resolve a path and ensure it stays inside the workspace.
    """

    workspace = get_workspace()
    workspace_root = workspace.resolve()
    target_path = (workspace / path).resolve()

    try:
        target_path.relative_to(workspace_root)
    except ValueError:
        return None

    return target_path

@tool
def search_code(query: str):
    """
    Search the workspace for a text string.
    """

    results = []

    workspace = get_workspace()

    if not workspace.exists():
        return {
            "error": "Workspace directory does not exist."
        }

    for file_path in workspace.rglob("*"):

        if not file_path.is_file():
            continue

        try:
            content = file_path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            if query.lower() in content.lower():
                results.append(str(file_path.relative_to(workspace)))

        except Exception:
            continue

    return {
        "query": query,
        "matches": results
    }
@tool
def read_file(path: str):
    """
    Read the contents of a file inside the workspace.
    """

    file_path = safe_workspace_path(path)

    if file_path is None:
        return {
            "error": "Access denied: path must stay inside the workspace."
        }

    if not file_path.exists():
        return {
            "error": "File does not exist."
        }

    if not file_path.is_file():
        return {
            "error": "The provided path is not a file."
        }

    try:
        content = file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        return {
            "path": path,
            "content": content
        }

    except Exception as e:
        return {
            "error": str(e)
        }
@tool
def list_dir(path: str = "."):
    """
    List files and directories inside the workspace.
    """

    directory = safe_workspace_path(path)

    if directory is None:
        return {
            "error": "Access denied: path must stay inside the workspace."
        }

    if not directory.exists():
        return {
            "error": "Directory does not exist."
        }

    if not directory.is_dir():
        return {
            "error": "The provided path is not a directory."
        }

    results = []

    for item in directory.rglob("*"):

        relative_path = item.relative_to(get_workspace())

        if item.is_dir():
            results.append(f"{relative_path}/")
        else:
            results.append(str(relative_path))

    return {
        "path": path,
        "items": results
    }

@tool
def propose_file_change(path: str, content: str):
    """
    Create a proposed file change without modifying the file.
    """

    file_path = safe_workspace_path(path)

    if file_path is None:
        return {
            "status": "error",
            "message": "Access denied: path must stay inside the workspace."
        }

    return {
        "status": "pending_approval",
        "path": path,
        "proposed_content": content,
        "message": "File change proposed. Human approval is required before writing."
    }

@tool
def git_diff(path: str, proposed_content: str):
    """
    Compare the current workspace file with proposed content.
    Does not modify the file.
    """

    file_path = safe_workspace_path(path)

    if file_path is None:
        return {
            "status": "error",
            "message": "Access denied: path must stay inside the workspace."
        }

    try:
        if file_path.exists():
            current_content = file_path.read_text(
                encoding="utf-8",
                errors="ignore"
            )
        else:
            current_content = ""

        diff = difflib.unified_diff(
            current_content.splitlines(keepends=True),
            proposed_content.splitlines(keepends=True),
            fromfile=f"current/{path}",
            tofile=f"proposed/{path}"
        )

        return {
            "status": "success",
            "path": path,
            "diff": "".join(diff)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
    
@tool
def write_file(path: str, content: str):
    """
    Create or overwrite a file inside the workspace.
    """

    file_path = safe_workspace_path(path)

    if file_path is None:
        return {
            "status": "error",
            "message": "Access denied: path must stay inside the workspace."
        }

    try:
        file_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        file_path.write_text(
            content,
            encoding="utf-8"
        )

        return {
            "path": path,
            "status": "success",
            "message": "File written successfully."
        }

    except Exception as e:
        return {
            "path": path,
            "status": "error",
            "message": str(e)
        }


@tool
def run_tests():
    """
    Run the repository's test/build command inside the active workspace.
    """
    workspace = get_workspace()

    if not workspace.exists():
        return {
            "status": "error",
            "message": "Workspace directory does not exist.",
        }

    import shutil as _shutil
    for cache_dir in workspace.rglob("__pycache__"):
        _shutil.rmtree(cache_dir, ignore_errors=True)

    try:
        if (workspace / "package.json").exists():
            import json

            package = json.loads(
                (workspace / "package.json").read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            )
            scripts = package.get("scripts", {})

            if "test" in scripts:
                command = ["npm.cmd", "test"]
            elif "build" in scripts:
                command = ["npm.cmd", "run", "build"]
            else:
                return {
                    "status": "error",
                    "message": "package.json has no test or build script.",
                }
        elif (workspace / "pyproject.toml").exists() or (workspace / "pytest.ini").exists() or list(workspace.glob("test*.py")):
            command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider"]
        else:
            return {
                "status": "error",
                "message": "Could not detect a supported test/build command.",
            }

        result = subprocess.run(
            command,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

        return {
            "status": "passed" if result.returncode == 0 else "failed",
            "exit_code": result.returncode,
            "command": command,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": "Test/build execution timed out after 120 seconds.",
        }
    except FileNotFoundError as e:
        return {
            "status": "error",
            "message": f"Required command was not found: {e}",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


if __name__ == "__main__":

    result = propose_file_change.invoke({
        "path": "auth.py",
        "content": "def login(username, password):\n    return authenticate(username, password)"
    })

    print(result)