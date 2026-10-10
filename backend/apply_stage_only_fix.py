"""
DevPilot fix: commit ONLY the files the agent wrote.

Problem: after `npm run build` the repo contains dist/ (and other build output),
and `git add -A` committed it into the pull request.

What this script does (in place, nothing else is touched):
  1. tools.py: write_file / write_new_file remember which files they wrote
     (stored inside .git/, so it is never committed).
  2. tools.py: git_commit stages only those files. If there is no record
     (older run) it stages everything EXCEPT common build folders.
  3. Adds backend/core/test_commit_staging.py (new file, 4 tests).

Safe to re-run. Your original tools.py is saved as tools.py.bak-before-staging.
Run from the repository root:   python apply_stage_only_fix.py
"""

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
TOOLS = ROOT / "backend" / "core" / "tools.py"
TEST = ROOT / "backend" / "core" / "test_commit_staging.py"

HELPERS = '''
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
        normalised = posixpath.normpath(str(path).replace("\\\\", "/"))
        files = _read_written(workspace)
        if normalised not in files:
            files.append(normalised)
        manifest.write_text(json.dumps(files), encoding="utf-8")
    except Exception:
        pass


def _stage_changes(workspace):
    written = _read_written(workspace)

    if written:
        _git(["add", "--", *written], workspace)
    else:
        # No record: stage everything except build output.
        _git(
            ["add", "-A", "--", ".", *[f":(exclude){d}" for d in BUILD_DIRS]],
            workspace,
        )


'''

TEST_FILE = '''import subprocess

import pytest

from core import tools as t


def _run(*args, cwd):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    _run("git", "init", "-b", "main", ".", cwd=work)
    (work / "README.md").write_text("# demo\\n")
    _run("git", "add", "-A", cwd=work)
    _run("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", "init", cwd=work)
    token = t.set_workspace(work)
    t.create_agent_branch.invoke({"branch_name": "devpilot/x"})
    yield work
    t.reset_workspace(token)


def test_commit_contains_only_files_the_agent_wrote(repo):
    assert t.write_new_file.invoke({"path": "src/theme.css", "content": ".dark {}\\n"})["status"] == "success"

    # Build output / scratch files created while testing must not be committed.
    (repo / "dist").mkdir()
    (repo / "dist" / "index.js").write_text("x")
    (repo / "notes.tmp").write_text("scratch")

    result = t.git_commit.invoke({"message": "m"})

    assert result["status"] == "success"
    assert result["files"] == ["src/theme.css"]


def test_edited_existing_file_is_recorded_too(repo):
    t.write_file.invoke({"path": "README.md", "content": "# changed\\n"})
    (repo / "dist").mkdir()
    (repo / "dist" / "a.js").write_text("x")

    assert t.git_commit.invoke({"message": "m"})["files"] == ["README.md"]


def test_without_a_record_common_build_folders_are_still_excluded(repo):
    (repo / "src").mkdir()
    (repo / "src" / "x.py").write_text("x = 1\\n")
    (repo / "dist").mkdir()
    (repo / "dist" / "bundle.js").write_text("x")
    (repo / "node_modules").mkdir()
    (repo / "node_modules" / "m.js").write_text("x")

    assert t.git_commit.invoke({"message": "m"})["files"] == ["src/x.py"]


def test_only_build_output_means_nothing_to_commit(repo):
    (repo / "dist").mkdir()
    (repo / "dist" / "bundle.js").write_text("x")

    assert t.git_commit.invoke({"message": "m"})["status"] == "nothing_to_commit"
'''


def fail(message):
    print("\nSTOPPED - nothing was changed: " + message)
    sys.exit(1)


if not TOOLS.is_file():
    fail(f"{TOOLS} not found. Run this from the DevPilot-AI folder.")

source = TOOLS.read_text(encoding="utf-8")

if "_stage_changes" in source:
    print("tools.py already contains the fix - nothing to change.")
else:
    patched = source

    # 1. helpers, placed before safe_workspace_path (exists in every version)
    anchor = patched.find("def safe_workspace_path(")
    if anchor == -1:
        fail("could not find 'def safe_workspace_path(' in tools.py")
    line_start = patched.rfind("\n", 0, anchor) + 1
    patched = patched[:line_start] + HELPERS.lstrip("\n") + "\n" + patched[line_start:]

    # 2. record paths after a successful write (write_file + write_new_file)
    pattern = re.compile(
        r'(file_path\.write_text\(\s*content,\s*encoding="utf-8"\s*\)\s*)'
        r'(return \{\s*"path":\s*path,\s*"status":\s*"success")'
    )
    patched, count = pattern.subn(r"\1_record_written(path)\n\n        \2", patched)
    if count != 2:
        fail(
            "expected to patch 2 write functions but matched %d. "
            "Your tools.py differs from what this script expects - send it to me." % count
        )

    # 3. stage only the recorded files
    old_stage = re.compile(
        r'_git\(\["add", "-A"\], workspace\)\s*'
        r'if not _git\(\["status", "--porcelain"\], workspace\):'
    )
    patched, count = old_stage.subn(
        '_stage_changes(workspace)\n\n'
        '        if not _git(["diff", "--cached", "--name-only"], workspace):',
        patched,
    )
    if count != 1:
        fail(
            "could not find the staging lines inside git_commit. "
            "Your tools.py differs from what this script expects - send it to me."
        )

    try:
        compile(patched, str(TOOLS), "exec")
    except SyntaxError as exc:
        fail("the patched file would not compile (%s); original left untouched." % exc)

    backup = TOOLS.with_name("tools.py.bak-before-staging")
    shutil.copy2(TOOLS, backup)
    TOOLS.write_text(patched, encoding="utf-8")
    print("tools.py patched.  Backup: %s  (delete it once everything works)" % backup.name)

if TEST.exists():
    print("test_commit_staging.py already exists - left as is.")
else:
    TEST.write_text(TEST_FILE, encoding="utf-8")
    print("Added backend/core/test_commit_staging.py")

print("\nNext:  cd backend   then   python -m pytest core/test_commit_staging.py")