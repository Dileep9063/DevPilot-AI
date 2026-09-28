from pathlib import Path
import difflib
from langchain_core.tools import tool
import os
import subprocess
import sys
WORKSPACE = Path("workspace")
def safe_workspace_path(path: str):
    """
    Resolve a path and ensure it stays inside the workspace.
    """

    workspace_root = WORKSPACE.resolve()
    target_path = (WORKSPACE / path).resolve()

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

    if not WORKSPACE.exists():
        return {
            "error": "Workspace directory does not exist."
        }

    for file_path in WORKSPACE.rglob("*"):

        if not file_path.is_file():
            continue

        try:
            content = file_path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            if query.lower() in content.lower():
                results.append(str(file_path))

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

    file_path = Path(path)

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

    directory = WORKSPACE / path

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

        relative_path = item.relative_to(WORKSPACE)

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
    Run the Python test suite inside the workspace.
    """

    if not WORKSPACE.exists():
        return {
            "status": "error",
            "message": "Workspace directory does not exist."
        }

    # Clear stale bytecode caches before every run. Without this, a
    # file rewritten with different content but the same size can be
    # served from a cached .pyc if the source mtime doesn't change at
    # a coarser-than-expected resolution, making the retry loop see a
    # PASS/FAIL result that doesn't match the actual current source.
    # This matters a lot here because write -> test -> debug -> fix ->
    # write -> test can happen multiple times within the same second.
    import shutil as _shutil

    for cache_dir in WORKSPACE.rglob("__pycache__"):
        _shutil.rmtree(cache_dir, ignore_errors=True)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-p", "no:cacheprovider"],
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        )

        return {
            "status": "passed" if result.returncode == 0 else "failed",
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": "Test execution timed out after 60 seconds."
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":

    result = propose_file_change.invoke({
        "path": "auth.py",
        "content": "def login(username, password):\n    return authenticate(username, password)"
    })

    print(result)