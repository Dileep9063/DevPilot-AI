import subprocess

import pytest

from core import tools as t


def _run(*args, cwd):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    _run("git", "init", "-b", "main", ".", cwd=work)
    (work / "README.md").write_text("# demo\n")
    _run("git", "add", "-A", cwd=work)
    _run("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", "init", cwd=work)
    token = t.set_workspace(work)
    t.create_agent_branch.invoke({"branch_name": "devpilot/x"})
    yield work
    t.reset_workspace(token)


def test_commit_contains_only_files_the_agent_wrote(repo):
    assert t.write_new_file.invoke({"path": "src/theme.css", "content": ".dark {}\n"})["status"] == "success"

    # Build output / scratch files created while testing must not be committed.
    (repo / "dist").mkdir()
    (repo / "dist" / "index.js").write_text("x")
    (repo / "notes.tmp").write_text("scratch")

    result = t.git_commit.invoke({"message": "m"})

    assert result["status"] == "success"
    assert result["files"] == ["src/theme.css"]


def test_edited_existing_file_is_recorded_too(repo):
    t.write_file.invoke({"path": "README.md", "content": "# changed\n"})
    (repo / "dist").mkdir()
    (repo / "dist" / "a.js").write_text("x")

    assert t.git_commit.invoke({"message": "m"})["files"] == ["README.md"]


def test_without_a_record_common_build_folders_are_still_excluded(repo):
    (repo / "src").mkdir()
    (repo / "src" / "x.py").write_text("x = 1\n")
    (repo / "dist").mkdir()
    (repo / "dist" / "bundle.js").write_text("x")
    (repo / "node_modules").mkdir()
    (repo / "node_modules" / "m.js").write_text("x")

    assert t.git_commit.invoke({"message": "m"})["files"] == ["src/x.py"]


def test_only_build_output_means_nothing_to_commit(repo):
    (repo / "dist").mkdir()
    (repo / "dist" / "bundle.js").write_text("x")

    assert t.git_commit.invoke({"message": "m"})["status"] == "nothing_to_commit"
