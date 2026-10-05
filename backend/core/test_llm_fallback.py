import pytest

from core import llm_providers as lp


class Fake:
    def __init__(self, name, fail=None, calls=None):
        self.name, self.fail, self.calls = name, fail, calls if calls is not None else []
        self.bound = False

    def bind_tools(self, tools):
        self.bound = True
        return self

    def invoke(self, messages):
        self.calls.append(self.name)
        if self.fail:
            raise Exception(self.fail)
        return f"ok-from-{self.name}"


NOSLEEP = lambda s: None


def test_first_provider_used_when_healthy():
    a, b = Fake("a"), Fake("b")
    out = lp.invoke_with_fallback(["hi"], providers=[("a", a), ("b", b)], sleep=NOSLEEP)
    assert out == "ok-from-a" and b.calls == []


def test_rate_limit_retries_then_falls_back():
    calls = []
    a = Fake("a", fail="429 rate limit", calls=calls)
    b = Fake("b", calls=calls)
    out = lp.invoke_with_fallback(
        ["hi"], providers=[("a", a), ("b", b)], retries=1, sleep=NOSLEEP
    )
    assert out == "ok-from-b"
    assert calls == ["a", "a", "b"]  # one retry on a, then b


def test_non_transient_error_skips_retry():
    calls = []
    a = Fake("a", fail="401 invalid api key", calls=calls)
    b = Fake("b", calls=calls)
    lp.invoke_with_fallback(["hi"], providers=[("a", a), ("b", b)], retries=3, sleep=NOSLEEP)
    assert calls == ["a", "b"]


def test_all_fail_reports_every_error():
    a, b = Fake("a", fail="boom A"), Fake("b", fail="boom B")
    with pytest.raises(lp.AllProvidersFailed) as e:
        lp.invoke_with_fallback(["hi"], providers=[("a", a), ("b", b)], retries=0, sleep=NOSLEEP)
    assert "boom A" in str(e.value) and "boom B" in str(e.value)


def test_tools_are_bound():
    a = Fake("a")
    lp.invoke_with_fallback(["hi"], tools=["t"], providers=[("a", a)], sleep=NOSLEEP)
    assert a.bound


def test_no_providers_configured(monkeypatch):
    with pytest.raises(lp.AllProvidersFailed):
        lp.invoke_with_fallback(["hi"], providers=[], sleep=NOSLEEP)


def test_provider_order_and_missing_keys(monkeypatch):
    for k in ("OPENROUTER_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "HF_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setitem(lp.FACTORIES, "openrouter", lambda: Fake("or"))
    monkeypatch.setitem(lp.FACTORIES, "gemini", lambda: Fake("gem"))
    monkeypatch.setitem(lp.FACTORIES, "huggingface", lambda: None)  # no token
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "gemini,openrouter,huggingface,bogus")
    names = [n for n, _ in lp.get_providers(refresh=True)]
    assert names == ["gemini", "openrouter"]


def test_transient_detection():
    assert lp.is_transient_provider_error(Exception("503 UNAVAILABLE"))
    assert lp.is_transient_provider_error(Exception("429 rate limit"))
    assert lp.is_transient_provider_error(Exception("quota exceeded"))
    assert not lp.is_transient_provider_error(Exception("invalid request"))
