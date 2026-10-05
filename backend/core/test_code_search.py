import pytest

from core import tools as t


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "auth.py").write_text(
        "def login(user, password):\n    return check(user, password)\n\n"
        "def logout(user):\n    return True\n"
    )
    (tmp_path / "views.py").write_text(
        "from auth import login\n\n"
        "def handle(request):\n    return login(request.user, request.pw)\n"
    )
    (tmp_path / "README.md").write_text("Notes. We mention login once.\n")
    (tmp_path / "big.py").write_text("x = 1\n" * 10)
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.py").write_text("def login(): pass\n")
    token = t.set_workspace(tmp_path)
    yield tmp_path
    t.reset_workspace(token)


def test_definition_outranks_usage(ws):
    r = t.search_code.invoke({"query": "login"})
    assert r["matches"][0] == "auth.py"
    assert "views.py" in r["matches"]


def test_snippets_have_line_numbers(ws):
    top = t.search_code.invoke({"query": "logout"})["results"][0]
    assert top["path"] == "auth.py"
    assert top["snippets"][0]["line"] == 4
    assert "def logout" in top["snippets"][0]["text"]


def test_ignored_directories_skipped(ws):
    assert all("node_modules" not in m for m in t.search_code.invoke({"query": "login"})["matches"])


def test_camel_and_snake_case_split(ws):
    (ws / "svc.py").write_text("def get_user_profile():\n    pass\n")
    r = t.search_code.invoke({"query": "userProfile"})
    assert "svc.py" in r["matches"]


def test_filename_match(ws):
    assert t.search_code.invoke({"query": "auth"})["matches"][0] == "auth.py"


def test_empty_and_no_match(ws):
    assert "error" in t.search_code.invoke({"query": "   "})
    assert t.search_code.invoke({"query": "zzzqqq"})["matches"] == []


def test_read_file_line_range(ws):
    r = t.read_file.invoke({"path": "big.py", "start_line": 2, "end_line": 4})
    assert r["start_line"] == 2 and r["end_line"] == 4 and r["total_lines"] == 10
    assert r["content"].count("\n") == 3
    assert "error" in t.read_file.invoke({"path": "big.py", "start_line": 50})
    assert t.read_file.invoke({"path": "big.py"})["content"].count("\n") == 10


def test_test_env_hides_secrets(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "s3cret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "s3cret")
    monkeypatch.setenv("LANGGRAPH_POSTGRES_URI", "postgresql://u:p@h/db")
    env = t._sanitized_env()
    assert not {"GITHUB_TOKEN", "OPENROUTER_API_KEY", "LANGGRAPH_POSTGRES_URI"} & set(env)
    assert "PATH" in env
