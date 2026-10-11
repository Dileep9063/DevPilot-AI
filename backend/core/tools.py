from pathlib import Path
import contextvars
import difflib
from langchain_core.tools import tool
import os
import subprocess
import sys
import time
import base64
import re
import requests

_NPM = "npm.cmd" if os.name == "nt" else "npm"


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
# GITHUB TOKEN CONTEXT
#
# In GitHub App mode each run gets a short-lived installation token
# (minted just before it is needed). It lives only in this context
# variable: never in the graph state, the database or .git/config.
# Without it, tools fall back to the single GITHUB_TOKEN from .env.
# ============================================================

_github_token_context = contextvars.ContextVar(
    "devpilot_github_token",
    default=None,
)


def set_github_token(token):
    return _github_token_context.set(token)


def reset_github_token(ctx_token):
    _github_token_context.reset(ctx_token)


def get_github_token():
    # Per-user GitHub App tokens always win. Production Compose disables the
    # shared fallback so one deployment token cannot write across user accounts.
    scoped_token = _github_token_context.get()
    if scoped_token:
        return scoped_token
    try:
        from django.conf import settings
        allow_shared = getattr(settings, "DEVPILOT_ALLOW_SHARED_GITHUB_TOKEN", True)
    except Exception:
        allow_shared = True  # standalone trusted local tools/tests
    return (os.environ.get("GITHUB_TOKEN") or "") if allow_shared else ""


def github_auth_args(token: str):
    """
    Arguments placed BEFORE the git subcommand (`git -c ... push`), so the
    credential applies to this one process and is never written to config.
    """
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return ["-c", f"http.extraheader=AUTHORIZATION: basic {basic}"]


# ============================================================
# GIT HELPERS
# ============================================================

def _git(args, cwd, timeout=60):
    """Run a git command; raises CalledProcessError on failure. Returns stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    return result.stdout.strip()


def _current_branch(cwd) -> str:
    """
    Name of the checked-out branch. Works on repositories that have no
    commits yet (unborn HEAD), where `rev-parse --abbrev-ref HEAD` fails.
    """
    try:
        return _git(["symbolic-ref", "--short", "HEAD"], cwd)
    except subprocess.CalledProcessError:
        return _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)  # detached HEAD


def _parse_github_remote(url: str):
    """Return (owner, repo) from a GitHub https/ssh remote URL, else None."""
    m = re.match(
        r"^(?:https://(?:[^@/]+@)?github\.com/|git@github\.com:)"
        r"([^/]+)/([^/]+?)(?:\.git)?/?$",
        url.strip(),
    )
    return m.groups() if m else None


def _redact(text: str) -> str:
    for token in {os.environ.get("GITHUB_TOKEN", ""), _github_token_context.get() or ""}:
        if token:
            text = text.replace(token, "***")
    return text


PROTECTED_BRANCHES = {"main", "master"}


_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "URI", "DATABASE_URL")


def _sanitized_env(workspace=None):
    """
    Reduced environment for repository tests.

    This is defense in depth, not a sandbox: untrusted code can still access
    files and resources available to the process. Run untrusted repos only in
    a separately isolated execution environment.
    """
    blocked_names = {
        "SSH_AUTH_SOCK", "SSH_AGENT_PID", "GIT_ASKPASS", "SSH_ASKPASS",
        "DOCKER_HOST", "DOCKER_CONFIG", "KUBECONFIG", "GPG_AGENT_INFO",
    }
    env = {
        k: v
        for k, v in os.environ.items()
        if k.upper() not in blocked_names
        and not any(m in k.upper() for m in _SECRET_MARKERS)
    }
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CI"] = "true"  # keep npm test runners out of watch mode
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    if workspace is not None:
        isolated_home = Path(workspace) / ".devpilot-home"
        isolated_home.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(isolated_home)
        env["XDG_CONFIG_HOME"] = str(isolated_home / ".config")
        env["XDG_CACHE_HOME"] = str(isolated_home / ".cache")
    return env


# ============================================================
# SAFE PATH HANDLING
# ============================================================

# ============================================================
# COMMIT ONLY WHAT THE AGENT WROTE
#
# Running the project's build/tests creates files such as dist/. They must
# never end up in the pull request, so write_file / write_new_file record
# each path (inside .git/, never committed) and git_commit stages just those.
# ============================================================

BUILD_DIRS = (
    "dist", "build", "node_modules", ".next", "out", "coverage",
    "__pycache__", ".pytest_cache", ".venv", "venv",
)


def _written_manifest(workspace):
    git_dir = Path(workspace) / ".git"
    return git_dir / "devpilot-written.json" if git_dir.is_dir() else None


def _read_written(workspace):
    import json

    manifest = _written_manifest(workspace)
    if manifest is None or not manifest.is_file():
        return []
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        return [p for p in data if isinstance(p, str)]
    except Exception:
        return []


def _record_written(path):
    """Remember that the agent wrote `path`. Bookkeeping must never break a write."""
    import json
    import posixpath

    try:
        workspace = get_workspace()
        manifest = _written_manifest(workspace)
        if manifest is None:
            return
        normalised = posixpath.normpath(str(path).replace("\\", "/"))
        files = _read_written(workspace)
        if normalised not in files:
            files.append(normalised)
        manifest.write_text(json.dumps(files), encoding="utf-8")
    except Exception:
        pass


def _stage_changes(workspace):
    written = _read_written(workspace)

    # Clear any pre-existing index entries first. Repository tests/build hooks
    # may run `git add`; never let those staged files sneak into an agent commit.
    # `git reset` needs HEAD, so use rm --cached for an unborn repository.
    try:
        _git(["rev-parse", "--verify", "HEAD"], workspace)
    except subprocess.CalledProcessError:
        _git(["rm", "--cached", "-r", "-f", "--ignore-unmatch", "."], workspace)
    else:
        _git(["reset", "--quiet"], workspace)
    if written:
        _git(["add", "--", *written], workspace)
    # Fail closed: never stage arbitrary repository changes when the agent
    # has not recorded an approved write. This prevents tests, build tools,
    # or pre-existing user changes from being committed accidentally.



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

MAX_SEARCH_FILE_BYTES = 500_000
MAX_SEARCH_FILES = 5000
MAX_RESULTS = 10
SNIPPETS_PER_FILE = 3

_DEFINITION_RE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?"
    r"(?:def|class|function|const|let|var|interface|type|struct|fn|func)\s+(\w+)"
)


SUBWORD_WEIGHT = 0.35


def _tokenize(text: str):
    """
    Return {token: weight}.

    The whole identifier (e.g. "diffview", "create_pull_request") has
    weight 1.0. Its snake_case/camelCase sub-words are also searched
    but weigh less, so "DiffView" does not match every file about "view".
    """
    weights = {}
    for word in re.findall(r"[A-Za-z0-9_]+", text):
        whole = word.lower().replace("_", "")
        parts = [
            p.lower()
            for p in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+", word)
        ]
        full = word.lower()
        if len(full) > 1:
            weights[full] = max(weights.get(full, 0), 1.0)
        for p in parts:
            if len(p) > 1 and p != full:
                w = 1.0 if len(parts) == 1 else SUBWORD_WEIGHT
                weights[p] = max(weights.get(p, 0), w)
    return weights


def rank_code_search(workspace, query: str):
    """
    Rank files for a query (BM25-style keyword scoring).

    Signals: term frequency (saturating), rarity of the term across the
    repo (IDF), matches in the file path, matches on definition lines
    (def/class/function/...), and an exact whole-query substring bonus.
    """
    import math

    workspace = Path(workspace)
    weights = _tokenize(query)
    terms = list(weights)
    phrase = query.strip().lower()

    if not terms:
        return []

    docs = []

    for count, file_path in enumerate(_iter_source_files(workspace)):
        if count >= MAX_SEARCH_FILES:
            break
        try:
            if file_path.stat().st_size > MAX_SEARCH_FILE_BYTES:
                continue
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        rel = str(file_path.relative_to(workspace)).replace("\\", "/")
        lines = text.splitlines()
        lower_lines = [ln.lower() for ln in lines]
        joined = "\n".join(lower_lines)

        tf = {t: joined.count(t) for t in terms}
        path_l = rel.lower()
        path_hits = sum(weights[t] for t in terms if t in path_l)

        if not any(tf.values()) and not path_hits:
            continue

        docs.append((rel, lines, lower_lines, tf, path_hits, phrase in joined))

    if not docs:
        return []

    n_docs = len(docs)
    df = {t: sum(1 for d in docs if d[3][t] > 0) for t in terms}
    avg_len = sum(len(d[1]) for d in docs) / n_docs or 1

    results = []

    for rel, lines, lower_lines, tf, path_hits, has_phrase in docs:
        k1, b = 1.4, 0.75
        score = 0.0

        for t in terms:
            if tf[t] == 0:
                continue
            idf = math.log(1 + (n_docs - df[t] + 0.5) / (df[t] + 0.5))
            norm = tf[t] * (k1 + 1) / (
                tf[t] + k1 * (1 - b + b * len(lines) / avg_len)
            )
            score += weights[t] * idf * norm

        score += 3.0 * path_hits

        # Filename equals the query (e.g. "auth" -> auth.py): strong signal.
        stem = Path(rel).stem.lower()
        if weights.get(stem, 0) >= 1.0:
            score += 8.0
        if has_phrase:
            score += 4.0

        # Score each line; keep the best few as snippets.
        line_scores = []
        definitions = 0
        for i, low in enumerate(lower_lines):
            hit = sum(weights[t] for t in terms if t in low)
            if not hit:
                continue
            ls = hit
            m = _DEFINITION_RE.match(lines[i])
            if m and any(
                weights[t] >= 1.0 and t in m.group(1).lower() for t in terms
            ):
                ls += 3
                definitions += 1
            line_scores.append((ls, i))

        # A file that DEFINES the thing outranks files that merely use it.
        score += min(definitions, 2) * 8.0

        line_scores.sort(key=lambda x: (-x[0], x[1]))
        snippets = [
            {"line": i + 1, "text": lines[i].strip()[:200]}
            for _, i in sorted(line_scores[:SNIPPETS_PER_FILE], key=lambda x: x[1])
        ]

        if not snippets and lines:
            snippets = [{"line": 1, "text": lines[0].strip()[:200]}]

        results.append({"path": rel, "score": round(score, 2), "snippets": snippets})

    results.sort(key=lambda r: (-r["score"], r["path"]))
    return results[:MAX_RESULTS]


@tool
def search_code(query: str):
    """
    Search the repository for relevant code and return the best-matching
    files, ranked by relevance, each with line numbers and snippets.

    Works with natural phrases, identifiers (camelCase / snake_case are
    split automatically) and filenames. Files that DEFINE a matching
    function/class rank higher than files that merely mention it.
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

    results = rank_code_search(workspace, query)

    return {
        "query": query,

        # Ranked file paths (kept for backward compatibility).
        "matches": [r["path"] for r in results],

        # Same files with scores and line snippets.
        "results": results,
    }


# ============================================================
# READ FILE
# ============================================================

@tool
def read_file(
    path: str,
    start_line: int = 0,
    end_line: int = 0
):
    """
    Read a file inside the workspace.

    Optionally pass start_line / end_line (1-based, inclusive) to read
    only part of a large file. Use the line numbers returned by
    search_code. Omit both to read the whole file.
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

        if start_line or end_line:

            lines = content.splitlines(keepends=True)
            total = len(lines)

            start = max(1, int(start_line or 1))
            end = min(total, int(end_line or total))

            if start > end:

                return {
                    "error":
                        f"Invalid range: file has {total} lines."
                }

            return {
                "path": path,
                "content": "".join(lines[start - 1:end]),
                "start_line": start,
                "end_line": end,
                "total_lines": total,
            }

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
        current = _current_branch(workspace)

        # Debug retries re-enter this tool; stay on the branch if already there.
        if current == branch_name:
            return {
                "status": "success",
                "branch": branch_name,
                "message": f"Already on branch '{branch_name}'.",
            }

        exists = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"refs/heads/{branch_name}"],
            cwd=workspace, capture_output=True, text=True, timeout=30,
        ).returncode == 0

        _git(["checkout"] + ([] if exists else ["-b"]) + [branch_name], workspace)

        return {
            "status": "success",
            "branch": branch_name,
            "message": f"Switched to branch '{branch_name}'.",
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
        return {"status": "error", "message": "Git branch creation timed out."}

    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# ============================================================
# GIT COMMIT / PUSH / PULL REQUEST
#
# These are called ONLY by LangGraph nodes (never bound to the LLM).
# ============================================================

@tool
def git_commit(message: str):
    """Stage all workspace changes and commit them on the current branch."""

    workspace = get_workspace()

    if not message or not message.strip():
        return {"status": "error", "message": "Commit message is required."}

    try:
        branch = _current_branch(workspace)

        if branch in PROTECTED_BRANCHES:
            return {
                "status": "error",
                "message": f"Refusing to commit directly to '{branch}'.",
            }

        _stage_changes(workspace)

        if not _git(["diff", "--cached", "--name-only"], workspace):
            return {"status": "nothing_to_commit", "branch": branch}

        _git(
            [
                "-c", "user.name=DevPilot AI",
                "-c", "user.email=devpilot-ai@users.noreply.github.com",
                "commit", "-m", message.strip(),
            ],
            workspace,
        )

        return {
            "status": "success",
            "branch": branch,
            "commit": _git(["rev-parse", "--short", "HEAD"], workspace),
            "files": _git(
                ["show", "--name-only", "--pretty=format:", "HEAD"], workspace
            ).splitlines(),
        }

    except subprocess.CalledProcessError as exc:
        return {
            "status": "error",
            "message": exc.stderr.strip() or exc.stdout.strip() or "Commit failed.",
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@tool
def git_push():
    """Push the current (non-protected) branch to origin."""

    workspace = get_workspace()

    try:
        branch = _current_branch(workspace)

        if branch in PROTECTED_BRANCHES:
            return {
                "status": "error",
                "message": f"Refusing to push protected branch '{branch}'.",
            }

        remote = _git(["remote", "get-url", "origin"], workspace)
        cmd = ["push", "-u", "origin", branch]

        if _parse_github_remote(remote):
            token = get_github_token()

            if not token:
                return {
                    "status": "error",
                    "message": (
                        "No GitHub credentials available: connect your GitHub "
                        "account, or set GITHUB_TOKEN; cannot push to GitHub."
                    ),
                }

            # Token is passed per-command only; it is never written to .git/config.
            cmd = github_auth_args(token) + cmd

        _git(cmd, workspace, timeout=180)

        return {"status": "success", "branch": branch}

    except subprocess.CalledProcessError as exc:
        return {
            "status": "error",
            "message": _redact(
                exc.stderr.strip() or exc.stdout.strip() or "Push failed."
            ),
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Git push timed out."}
    except Exception as exc:
        return {"status": "error", "message": _redact(str(exc))}


@tool
def create_pull_request(title: str, body: str = ""):
    """Open a GitHub pull request from the current branch into the default branch."""

    workspace = get_workspace()
    token = get_github_token()

    if not token:
        return {
            "status": "error",
            "message": "No GitHub credentials available (connect GitHub or set GITHUB_TOKEN).",
        }

    try:
        head = _current_branch(workspace)
        parsed = _parse_github_remote(_git(["remote", "get-url", "origin"], workspace))

        if not parsed:
            return {"status": "error", "message": "origin is not a GitHub remote."}

        owner, repo = parsed
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        api = f"https://api.github.com/repos/{owner}/{repo}"

        info = requests.get(api, headers=headers, timeout=30)
        info.raise_for_status()
        base = info.json()["default_branch"]

        if head == base:
            return {"status": "error", "message": "Head and base branch are the same."}

        resp = requests.post(
            f"{api}/pulls",
            headers=headers,
            json={"title": title, "body": body, "head": head, "base": base},
            timeout=30,
        )

        if resp.status_code == 422 and "already exists" in resp.text:
            existing = requests.get(
                f"{api}/pulls",
                headers=headers,
                params={"head": f"{owner}:{head}", "state": "open"},
                timeout=30,
            ).json()
            if existing:
                return {
                    "status": "success",
                    "url": existing[0]["html_url"],
                    "number": existing[0]["number"],
                    "note": "Pull request already existed.",
                }

        resp.raise_for_status()
        data = resp.json()

        return {"status": "success", "url": data["html_url"], "number": data["number"]}

    except requests.RequestException as exc:
        return {"status": "error", "message": _redact(str(exc))}
    except Exception as exc:
        return {"status": "error", "message": _redact(str(exc))}


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

        _record_written(path)

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

        _record_written(path)

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

_MAX_TEST_OUTPUT_CHARS = 20000


def _build_test_command(command, workspace, mode=None):
    """Build the requested test command, optionally inside bubblewrap.

    bubblewrap mode mounts only system runtime paths and the active workspace,
    drops the inherited environment, and disables network access. If the
    sandbox prerequisites are missing, fail closed rather than falling back.
    """
    import shutil

    mode = (mode or os.environ.get("DEVPILOT_TEST_SANDBOX", "unsafe-local")).lower()
    if mode == "unsafe-local":
        return list(command)
    if mode != "bubblewrap":
        raise ValueError(
            "Test execution is disabled. Set DEVPILOT_TEST_SANDBOX=bubblewrap "
            "or explicitly choose unsafe-local for trusted local development."
        )

    bwrap = shutil.which("bwrap")
    prlimit = shutil.which("prlimit")
    if not bwrap or not prlimit:
        raise FileNotFoundError(
            "Sandbox prerequisites are missing (bubblewrap and prlimit). "
            "Refusing to execute repository tests without isolation."
        )

    workspace = Path(workspace).resolve()
    sandbox_workspace = "/tmp/workspace"
    try:
        tmpfs_bytes = int(os.environ.get("DEVPILOT_TEST_TMPFS_BYTES", 512 * 1024 * 1024))
    except (TypeError, ValueError):
        tmpfs_bytes = 512 * 1024 * 1024
    # Clamp to a safe finite range so the sandbox's writable tmpfs is bounded.
    tmpfs_bytes = max(64 * 1024 * 1024, min(tmpfs_bytes, 2 * 1024 * 1024 * 1024))
    args = [
        prlimit,
        "--cpu=120:120",
        "--as=6442450944:6442450944",
        "--fsize=104857600:104857600",
        "--nofile=256:256",
        "--nproc=128:128",
        "--",
        bwrap,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--size", str(tmpfs_bytes),
    ]
    # Mount the runtime read-only; the app process runs as an unprivileged user.
    for system_path in ("/usr", "/etc", "/bin", "/sbin", "/lib", "/lib64", "/app", "/opt"):
        if Path(system_path).exists():
            args.extend(["--ro-bind", system_path, system_path])

    args.extend([
        "--tmpfs", "/tmp",
        "--dir", sandbox_workspace,
        "--bind", str(workspace), sandbox_workspace,
    ])
    # Hide other runs and common per-user/runtime write locations. The active
    # workspace is already bind-mounted elsewhere before /data is hidden.
    for hidden_path in ("/data", "/home", "/root", "/run", "/var/tmp", "/mnt", "/media", "/srv"):
        if Path(hidden_path).exists():
            args.extend(["--tmpfs", hidden_path])

    args.extend([
        "--proc", "/proc",
        "--dev", "/dev",
        "--chdir", sandbox_workspace,
        "--clearenv",
        "--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin",
        "--setenv", "HOME", sandbox_workspace + "/.devpilot-home",
        "--setenv", "XDG_CACHE_HOME", sandbox_workspace + "/.devpilot-home/.cache",
        "--setenv", "CI", "true",
        "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
        "--",
        *command,
    ])
    return args


_MAX_CAPTURE_BYTES = 80000


def _run_bounded_subprocess(
    command,
    cwd,
    env,
    timeout=120,
    max_output_bytes=_MAX_CAPTURE_BYTES,
    max_workspace_bytes=None,
    max_workspace_files=None,
):
    """Run a command with bounded output and best-effort workspace growth limits.

    Workspace limits are monitored periodically and are not a kernel-enforced
    disk quota. Per-file limits in bubblewrap mode provide an additional guard.
    """
    import signal
    import threading

    max_workspace_bytes = max_workspace_bytes or int(
        os.environ.get("DEVPILOT_TEST_MAX_WORKSPACE_BYTES", 500 * 1024 * 1024)
    )
    max_workspace_files = max_workspace_files or int(
        os.environ.get("DEVPILOT_TEST_MAX_WORKSPACE_FILES", 50000)
    )

    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        start_new_session=(os.name != "nt"),
    )
    buffers = {"stdout": [], "stderr": []}
    counts = {"stdout": 0, "stderr": 0}
    truncated = {"stdout": False, "stderr": False}

    def drain(name, pipe):
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                break
            remaining = max_output_bytes - counts[name]
            if remaining > 0:
                kept = chunk[:remaining]
                buffers[name].append(kept)
                counts[name] += len(kept)
            if len(chunk) > max(remaining, 0):
                truncated[name] = True

    readers = [
        threading.Thread(target=drain, args=("stdout", process.stdout), daemon=True),
        threading.Thread(target=drain, args=("stderr", process.stderr), daemon=True),
    ]
    for reader in readers:
        reader.start()

    timed_out = False
    workspace_limit_exceeded = threading.Event()
    workspace_root = Path(cwd).resolve()

    def monitor_workspace():
        while process.poll() is None:
            total_bytes = 0
            total_files = 0
            try:
                for root, dirs, files in os.walk(workspace_root, followlinks=False):
                    dirs[:] = [
                        name for name in dirs
                        if not (Path(root) / name).is_symlink()
                    ]
                    for name in files:
                        path = Path(root) / name
                        try:
                            if path.is_symlink():
                                continue
                            stat_result = path.stat(follow_symlinks=False)
                        except OSError:
                            continue
                        if not path.is_file():
                            continue
                        total_files += 1
                        total_bytes += stat_result.st_size
                        if total_bytes > max_workspace_bytes or total_files > max_workspace_files:
                            workspace_limit_exceeded.set()
                            try:
                                if os.name != "nt":
                                    os.killpg(process.pid, signal.SIGKILL)
                                else:
                                    process.kill()
                            except ProcessLookupError:
                                pass
                            return
            except OSError:
                pass
            time.sleep(0.5)

    monitor = threading.Thread(target=monitor_workspace, daemon=True)
    monitor.start()
    try:
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        return_code = process.wait()
    monitor.join(timeout=1)

    for reader in readers:
        reader.join(timeout=2)
    for pipe in (process.stdout, process.stderr):
        if pipe:
            pipe.close()

    def output(name):
        value = b"".join(buffers[name]).decode("utf-8", errors="replace")
        if truncated[name]:
            value += "\n...[capture truncated]"
        return value

    stdout, stderr = output("stdout"), output("stderr")
    if workspace_limit_exceeded.is_set():
        raise RuntimeError(
            "Repository test stopped because workspace output exceeded the configured "
            "file-count or disk-usage limit."
        )
    if timed_out:
        raise subprocess.TimeoutExpired(
            command, timeout, output=stdout, stderr=stderr
        )
    return subprocess.CompletedProcess(command, return_code, stdout, stderr)


def _bounded_output(value):
    """Decode and cap command output before storing it in agent state."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    value = str(value)
    if len(value) <= _MAX_TEST_OUTPUT_CHARS:
        return value
    omitted = len(value) - _MAX_TEST_OUTPUT_CHARS
    return value[:_MAX_TEST_OUTPUT_CHARS] + f"\n...[truncated {omitted} characters]"
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
                    _NPM,
                    "test"
                ]

            elif "build" in scripts:

                command = [
                    _NPM,
                    "run",
                    "build"
                ]

            else:

                return {
                    "status":
                        "no_tests",

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

            # Nothing to run is not a failure: the change is simply unverified.
            return {
                "status":
                    "no_tests",

                "message":
                    "No tests were found, so this change was not verified.",
            }

        # ----------------------------------------------------
        # Execute
        # ----------------------------------------------------

        execution_command = _build_test_command(command, workspace)
        if os.environ.get("DEVPILOT_TEST_SANDBOX", "unsafe-local").lower() == "unsafe-local":
            import logging
            logging.getLogger(__name__).warning(
                "Running repository tests without a sandbox; use bubblewrap mode for untrusted code."
            )
        result = _run_bounded_subprocess(
            execution_command,
            cwd=workspace,
            env=_sanitized_env(workspace),
            timeout=120,
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

            "stdout": _bounded_output(result.stdout),

            "stderr": _bounded_output(result.stderr),
        }

    except subprocess.TimeoutExpired as exc:
        return {
            "status": "error",
            "message": "Test/build execution timed out after 120 seconds.",
            "stdout": _bounded_output(exc.stdout),
            "stderr": _bounded_output(exc.stderr),
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
