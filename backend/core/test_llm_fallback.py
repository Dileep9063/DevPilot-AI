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
    assert "a: Exception" in str(e.value) and "b: Exception" in str(e.value)
    assert "boom A" not in str(e.value) and "boom B" not in str(e.value)


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



def test_usage_logging_includes_tokens_and_configured_cost(monkeypatch, caplog):
    class UsageResult:
        usage_metadata = {"input_tokens": 1000, "output_tokens": 500}
        response_metadata = {"model_name": "test-model"}

    class UsageFake(Fake):
        def invoke(self, messages):
            self.calls.append(self.name)
            return UsageResult()

    monkeypatch.setenv("LLM_COST_TEST_INPUT_PER_1M", "2")
    monkeypatch.setenv("LLM_COST_TEST_OUTPUT_PER_1M", "4")
    fake = UsageFake("test")

    with caplog.at_level("INFO", logger="core.llm_providers"):
        lp.invoke_with_fallback(
            ["hello"], providers=[("test", fake)], retries=0, sleep=NOSLEEP
        )

    line = next(record.getMessage() for record in caplog.records if "LLM usage:" in record.getMessage())
    assert "input_tokens=1000" in line
    assert "output_tokens=500" in line
    assert "estimated_cost_usd=0.004" in line
    assert "test-model" in line


def test_usage_logging_handles_missing_token_metadata(caplog):
    with caplog.at_level("INFO", logger="core.llm_providers"):
        lp.invoke_with_fallback(
            ["hello"], providers=[("test", Fake("test"))], retries=0, sleep=NOSLEEP
        )
    line = next(record.getMessage() for record in caplog.records if "LLM usage:" in record.getMessage())
    assert "input_tokens=unknown" in line
    assert "output_tokens=unknown" in line



def test_cost_is_unknown_when_provider_rates_are_not_configured(monkeypatch, caplog):
    class UsageResult:
        usage_metadata = {"input_tokens": 100, "output_tokens": 50}
        response_metadata = {"model_name": "test-model"}

    class UsageFake(Fake):
        def invoke(self, messages):
            self.calls.append(self.name)
            return UsageResult()

    monkeypatch.delenv("LLM_COST_TEST_INPUT_PER_1M", raising=False)
    monkeypatch.delenv("LLM_COST_TEST_OUTPUT_PER_1M", raising=False)
    with caplog.at_level("INFO", logger="core.llm_providers"):
        lp.invoke_with_fallback(
            ["hello"], providers=[("test", UsageFake("test"))], retries=0, sleep=NOSLEEP
        )
    line = next(record.getMessage() for record in caplog.records if "LLM usage:" in record.getMessage())
    assert "input_tokens=100" in line
    assert "output_tokens=50" in line
    assert "estimated_cost_usd=unknown" in line
