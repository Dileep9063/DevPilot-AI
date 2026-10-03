from pathlib import Path
import contextvars
import difflib
from langchain_core.tools import tool
import os
import subprocess
import sys


# ============================================================
# WORKSPACE CONTEXT
# ============================================================

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


# ============================================================
# SAFE PATH HANDLING
# ============================================================

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


# ============================================================
# SEARCH CONFIGURATION
# ============================================================

IGNORED_DIRECTORIES = {
    ".git",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".venv",
    "venv",
    "coverage",
    ".next",
    ".cache",
    ".pytest_cache",
}


SEARCHABLE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".css",
    ".scss",
    ".html",
    ".json",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".env.example",
}


def _iter_source_files(root):
    """
    Yield relevant source/config files while skipping
    dependencies and generated trees.
    """

    for current_root, dirs, files in os.walk(root):

        dirs[:] = [
            d
            for d in dirs
            if d not in IGNORED_DIRECTORIES
        ]

        for filename in files:

            path = Path(current_root) / filename

            if (
                path.suffix.lower()
                in SEARCHABLE_EXTENSIONS
                or filename in {
                    "Dockerfile",
                    "Makefile",
                }
            ):

                yield path


# ============================================================
# SEARCH CODE
# ============================================================

@tool
def search_code(query: str):
    """
    Search relevant source/config files while ignoring
    dependencies and generated files.
    """

    workspace = get_workspace()

    if not workspace.exists():

        return {
            "error":
                "Workspace directory does not exist."
        }

    query = query.strip()

    if not query:

        return {
            "error":
                "Search query cannot be empty."
        }

    results = []

    for file_path in _iter_source_files(workspace):

        try:

            content = file_path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            if query.lower() in content.lower():

                results.append(
                    str(
                        file_path.relative_to(
                            workspace
                        )
                    )
                )

        except Exception:

            continue

    return {
        "query": query,
        "matches": results
    }


# ============================================================
# READ FILE
# ============================================================

@tool
def read_file(path: str):
    """
    Read the contents of a file inside the workspace.
    """

    file_path = safe_workspace_path(path)

    if file_path is None:

        return {
            "error":
                "Access denied: path must stay inside the workspace."
        }

    if not file_path.exists():

        return {
            "error":
                "File does not exist."
        }

    if not file_path.is_file():

        return {
            "error":
                "The provided path is not a file."
        }

    try:

        content = file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        return {
            "path":
                path,

            "content":
                content
        }

    except Exception as e:

        return {
            "error":
                str(e)
        }


# ============================================================
# LIST DIRECTORY
# ============================================================

@tool
def list_dir(path: str = "."):
    """
    List files and directories inside the workspace.
    """

    directory = safe_workspace_path(path)

    if directory is None:

        return {
            "error":
                "Access denied: path must stay inside the workspace."
        }

    if not directory.exists():

        return {
            "error":
                "Directory does not exist."
        }

    if not directory.is_dir():

        return {
            "error":
                "The provided path is not a directory."
        }

    results = []

    workspace = get_workspace()

    # Recursively list only relevant project files/directories.
    # Dependency, VCS, cache, and generated trees are hidden.

    for current_root, dirs, files in os.walk(directory):

        dirs[:] = [
            d
            for d in dirs
            if d not in IGNORED_DIRECTORIES
        ]

        current_path = Path(current_root)

        if current_path != directory:

            results.append(
                f"{current_path.relative_to(workspace)}/"
            )

        for filename in files:

            file_path = current_path / filename

            relative_path = (
                file_path.relative_to(workspace)
            )

            results.append(
                str(relative_path)
            )

    return {
        "path":
            path,

        "items":
            results
    }


# ============================================================
# PROPOSE EXISTING FILE CHANGE
# ============================================================

@tool
def propose_file_change(
    path: str,
    old_text: str,
    new_text: str
):
    """
    Propose a targeted text replacement inside an existing file.

    The tool does NOT modify the file.

    Args:
        path:
            Relative path to the existing file.

        old_text:
            Exact existing text that should be replaced.

        new_text:
            Replacement text.
    """

    if not path or not path.strip():

        return {
            "status":
                "error",

            "message":
                "A valid relative file path is required."
        }

    if (
        not isinstance(old_text, str)
        or not isinstance(new_text, str)
    ):

        return {
            "status":
                "error",

            "message":
                "old_text and new_text must be strings."
        }

    file_path = safe_workspace_path(path)

    if file_path is None:

        return {
            "status":
                "error",

            "message":
                "Access denied: path must stay inside the workspace."
        }

    if not file_path.exists():

        return {
            "status":
                "error",

            "message":
                f"File does not exist: {path}"
        }

    if not file_path.is_file():

        return {
            "status":
                "error",

            "message":
                f"Path is not a file: {path}"
        }

    try:

        current_content = file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    except Exception as e:

        return {
            "status":
                "error",

            "message":
                f"Unable to read file: {e}"
        }

    if not old_text:

        return {
            "status":
                "error",

            "message":
                "old_text cannot be empty."
        }

    occurrences = current_content.count(
        old_text
    )

    if occurrences == 0:

        return {
            "status":
                "error",

            "message": (
                "The supplied old_text was not found in the file. "
                "Read the file again and provide an exact "
                "existing section."
            )
        }

    if occurrences > 1:

        return {
            "status":
                "error",

            "message": (
                f"old_text occurs {occurrences} times. "
                "Provide a more specific text section."
            )
        }

    proposed_content = current_content.replace(
        old_text,
        new_text,
        1
    )

    # Prevent meaningless proposals.
    if proposed_content == current_content:

        return {
            "status":
                "error",

            "message":
                "Proposed change would not modify the file."
        }

    return {
        "status":
            "pending_approval",

        "path":
            path,

        "proposed_content":
            proposed_content,

        "message": (
            "Targeted file change proposed successfully. "
            "The file has NOT been modified. "
            "Human approval is required before writing."
        )
    }


# ============================================================
# PROPOSE NEW FILE
# ============================================================

@tool
def propose_new_file(
    path: str,
    content: str
) -> dict:
    """
    Propose creating a new file.

    The file is NOT created immediately.
    A human approval is required before writing.
    """

    if not path or not path.strip():

        return {
            "status":
                "error",

            "message":
                "A valid relative file path is required.",
        }

    if not isinstance(content, str):

        return {
            "status":
                "error",

            "message":
                "File content must be a string.",
        }

    target = safe_workspace_path(path)

    if target is None:

        return {
            "status":
                "error",

            "message":
                "Access denied: path must stay inside the workspace.",
        }

    if target.exists():

        return {
            "status":
                "error",

            "message": (
                f"File already exists: {path}. "
                "Use propose_file_change instead."
            ),
        }

    if not content.strip():

        return {
            "status":
                "error",

            "message":
                "Cannot create an empty file.",
        }

    return {
        "status":
            "pending_new_file_approval",

        "path":
            path,

        "proposed_content":
            content,

        "message": (
            "New file creation proposed successfully. "
            "The file has NOT been created. "
            "Human approval is required before writing."
        ),
    }


# ============================================================
# GIT DIFF
# ============================================================

@tool
def git_diff(
    path: str,
    proposed_content: str
):
    """
    Compare the current workspace file with proposed content.

    Does not modify the file.
    """

    file_path = safe_workspace_path(path)

    if file_path is None:

        return {
            "status":
                "error",

            "message":
                "Access denied: path must stay inside the workspace."
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
            current_content.splitlines(
                keepends=True
            ),

            proposed_content.splitlines(
                keepends=True
            ),

            fromfile=f"current/{path}",

            tofile=f"proposed/{path}"
        )

        return {
            "status":
                "success",

            "path":
                path,

            "diff":
                "".join(diff)
        }

    except Exception as e:

        return {
            "status":
                "error",

            "message":
                str(e)
        }
@tool
def git_status():
    """
    Return the current Git working-tree status
    for the active workspace.
    """

    workspace = get_workspace()

    if workspace is None:
        return {
            "status": "error",
            "message": "Workspace is not initialized."
        }

    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )

        return {
            "status": "success",
            "changes": result.stdout.strip(),
        }

    except subprocess.CalledProcessError as exc:
        return {
            "status": "error",
            "message": (
                exc.stderr.strip()
                or exc.stdout.strip()
                or "Unable to read Git status."
            ),
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": "Git status timed out."
        }

    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
        }
@tool
def git_status():
    """
    Return the current Git working-tree status
    for the active workspace.
    """

    workspace = get_workspace()

    if workspace is None:
        return {
            "status": "error",
            "message": "Workspace is not initialized."
        }

    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )

        return {
            "status": "success",
            "changes": result.stdout.strip(),
        }

    except subprocess.CalledProcessError as exc:
        return {
            "status": "error",
            "message": (
                exc.stderr.strip()
                or exc.stdout.strip()
                or "Unable to read Git status."
            ),
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": "Git status timed out."
        }

    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
        }

@tool
def create_agent_branch(branch_name: str):
    """
    Create and switch to a dedicated Git branch
    for the current DevPilot run.
    """

    workspace = get_workspace()

    if workspace is None:
        return {
            "status": "error",
            "message": "Workspace is not initialized."
        }

    if not branch_name:
        return {
            "status": "error",
            "message": "Branch name is required."
        }

    if branch_name in {"main", "master"}:
        return {
            "status": "error",
            "message": "DevPilot cannot use the main/master branch."
        }

    try:
        result = subprocess.run(
            [
                "git",
                "checkout",
                "-b",
                branch_name,
            ],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )

        return {
            "status": "success",
            "branch": branch_name,
            "message": (
                result.stdout.strip()
                or f"Created and switched to branch '{branch_name}'."
            ),
        }

    except subprocess.CalledProcessError as exc:
        return {
            "status": "error",
            "message": (
                exc.stderr.strip()
                or exc.stdout.strip()
                or "Unable to create Git branch."
            ),
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": "Git branch creation timed out."
        }

    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
        }
# ============================================================
# WRITE EXISTING FILE
# ============================================================

@tool
def write_file(
    path: str,
    content: str
):
    """
    Create or overwrite a file inside the workspace.

    This function is intended for approved existing-file changes.
    """

    if not path or not path.strip():

        return {
            "status":
                "error",

            "message":
                "A valid relative file path is required."
        }

    file_path = safe_workspace_path(path)

    if file_path is None:

        return {
            "status":
                "error",

            "message":
                "Access denied: path must stay inside the workspace."
        }

    if file_path.exists() and not file_path.is_file():

        return {
            "status":
                "error",

            "message":
                "The provided path is not a file."
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
            "path":
                path,

            "status":
                "success",

            "message":
                "File written successfully."
        }

    except Exception as e:

        return {
            "path":
                path,

            "status":
                "error",

            "message":
                str(e)
        }


# ============================================================
# WRITE NEW FILE
# ============================================================

@tool
def write_new_file(
    path: str,
    content: str
):
    """
    Create a NEW file inside the workspace.

    This function should only be called by the graph after
    human approval has been granted.
    """

    if not path or not path.strip():

        return {
            "status":
                "error",

            "message":
                "A valid relative file path is required."
        }

    if not isinstance(content, str):

        return {
            "status":
                "error",

            "message":
                "File content must be a string."
        }

    file_path = safe_workspace_path(path)

    if file_path is None:

        return {
            "status":
                "error",

            "message":
                "Access denied: path must stay inside the workspace."
        }

    if file_path.exists():

        return {
            "status":
                "error",

            "message": (
                f"File already exists: {path}. "
                "Use write_file for an existing file."
            )
        }

    if not content.strip():

        return {
            "status":
                "error",

            "message":
                "Cannot create an empty file."
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
            "path":
                path,

            "status":
                "success",

            "message":
                "New file created successfully."
        }

    except Exception as e:

        return {
            "path":
                path,

            "status":
                "error",

            "message":
                str(e)
        }


# ============================================================
# RUN TESTS
# ============================================================

@tool
def run_tests():
    """
    Run the repository's test/build command inside the active workspace.
    """

    workspace = get_workspace()

    if not workspace.exists():

        return {
            "status":
                "error",

            "message":
                "Workspace directory does not exist.",
        }

    import shutil as _shutil

    for cache_dir in workspace.rglob(
        "__pycache__"
    ):

        _shutil.rmtree(
            cache_dir,
            ignore_errors=True
        )

    try:

        # ----------------------------------------------------
        # JavaScript / Node project
        # ----------------------------------------------------

        if (workspace / "package.json").exists():

            import json

            package = json.loads(
                (
                    workspace / "package.json"
                ).read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            )

            scripts = package.get(
                "scripts",
                {}
            )

            if "test" in scripts:

                command = [
                    "npm.cmd",
                    "test"
                ]

            elif "build" in scripts:

                command = [
                    "npm.cmd",
                    "run",
                    "build"
                ]

            else:

                return {
                    "status":
                        "error",

                    "message":
                        "package.json has no test or build script.",
                }

        # ----------------------------------------------------
        # Python project
        # ----------------------------------------------------

        elif (
            (workspace / "pyproject.toml").exists()
            or (workspace / "pytest.ini").exists()
            or list(workspace.glob("test*.py"))
        ):

            command = [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "no:cacheprovider"
            ]

        else:

            return {
                "status":
                    "error",

                "message":
                    "Could not detect a supported test/build command.",
            }

        # ----------------------------------------------------
        # Execute
        # ----------------------------------------------------

        result = subprocess.run(
            command,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=120,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE":
                    "1",
            },
        )

        return {
            "status":
                "passed"
                if result.returncode == 0
                else "failed",

            "exit_code":
                result.returncode,

            "command":
                command,

            "stdout":
                result.stdout,

            "stderr":
                result.stderr,
        }

    except subprocess.TimeoutExpired:

        return {
            "status":
                "error",

            "message":
                "Test/build execution timed out after 120 seconds.",
        }

    except FileNotFoundError as e:

        return {
            "status":
                "error",

            "message":
                f"Required command was not found: {e}",
        }

    except Exception as e:

        return {
            "status":
                "error",

            "message":
                str(e),
        }


# ============================================================
# DIRECT TOOL TEST
# ============================================================

if __name__ == "__main__":

    print(
        "DevPilot AI tools loaded successfully."
    )
