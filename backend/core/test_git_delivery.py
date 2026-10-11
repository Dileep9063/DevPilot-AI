import subprocess

import pytest

from core import tools as t


def _run(*args, cwd):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    remote = tmp_path / "remote.git"
    work = tmp_path / "work"
    _run("git", "init", "--bare", str(remote), cwd=tmp_path)
    _run("git", "clone", str(remote), str(work), cwd=tmp_path)
    _run("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit",
         "--allow-empty", "-m", "init", cwd=work)
    _run("git", "branch", "-M", "main", cwd=work)
    token = t.set_workspace(work)
    yield work
    t.reset_workspace(token)


def test_branch_creation_is_idempotent(repo):
    assert t.create_agent_branch.invoke({"branch_name": "devpilot/x"})["status"] == "success"
    assert t.create_agent_branch.invoke({"branch_name": "devpilot/x"})["status"] == "success"


def test_main_is_protected(repo):
    assert t.create_agent_branch.invoke({"branch_name": "main"})["status"] == "error"
    assert t.git_commit.invoke({"message": "m"})["status"] == "error"
    assert t.git_push.invoke({})["status"] == "error"


def test_commit_and_push(repo):
    t.create_agent_branch.invoke({"branch_name": "devpilot/y"})
    (repo / "a.py").write_text("x = 1\n")
    t._record_written("a.py")
    assert t.git_commit.invoke({"message": "add a"})["status"] == "success"
    assert t.git_commit.invoke({"message": "again"})["status"] == "nothing_to_commit"
    assert t.git_push.invoke({})["status"] == "success"


def test_pr_requires_token(repo, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert t.create_pull_request.invoke({"title": "t"})["status"] == "error"


def test_parse_github_remote():
    assert t._parse_github_remote("https://github.com/o/r.git") == ("o", "r")
    assert t._parse_github_remote("git@github.com:o/r.git") == ("o", "r")
    assert t._parse_github_remote("/tmp/local.git") is None


def test_branch_and_commit_work_on_repo_with_no_commits(tmp_path):
    """Regression: a brand-new repo (unborn HEAD) must not break branch creation."""
    _run("git", "init", "-b", "main", str(tmp_path), cwd=tmp_path)
    token = t.set_workspace(tmp_path)
    try:
        assert t.create_agent_branch.invoke({"branch_name": "devpilot/new"})["status"] == "success"
        (tmp_path / "a.py").write_text("x = 1\n")
        t._record_written("a.py")
        assert t.git_commit.invoke({"message": "first commit"})["status"] == "success"
    finally:
        t.reset_workspace(token)


def test_push_to_github_remote_without_token_gives_clear_error(repo, monkeypatch):
    """The exact failure seen in practice: GITHUB_TOKEN missing from the environment."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    _run("git", "remote", "set-url", "origin", "https://github.com/o/r.git", cwd=repo)
    t.create_agent_branch.invoke({"branch_name": "devpilot/z"})
    (repo / "a.py").write_text("x = 1\n")
    t._record_written("a.py")
    t.git_commit.invoke({"message": "m"})

    result = t.git_push.invoke({})

    assert result["status"] == "error"
    assert "No GitHub credentials" in result["message"]
