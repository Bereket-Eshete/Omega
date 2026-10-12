"""Shared local inference must exclude concurrent tokenizer/model access."""

import importlib.util
import sys
import threading
import types
from pathlib import Path

import pytest

TIMEOUT = 5
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def llm(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    openai = types.ModuleType("openai")
    openai.OpenAI = object
    monkeypatch.setitem(sys.modules, "openai", openai)
    spec = importlib.util.spec_from_file_location(
        "local_embedding_test", REPO_ROOT / "providers/lib_llm_ext.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ObservedLock:
    """Signal the second acquisition attempt before it waits on a real lock."""

    def __init__(self):
        self.lock = threading.Lock()
        self.attempts = 0
        self.second_attempt = threading.Event()

    def __enter__(self):
        self.attempts += 1
        if self.attempts == 2:
            self.second_attempt.set()
        self.lock.acquire()

    def __exit__(self, *args):
        self.lock.release()


def test_local_encodes_do_not_overlap(llm, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    lock = ObservedLock()
    monkeypatch.setattr(llm, "_embedding_lock", lock)
    calls, results, errors = [], [], []

    class Vector:
        def tolist(self):
            return [1.0, 0.0]

    class Model:
        def encode(self, text, normalize_embeddings):
            calls.append(text)
            assert normalize_embeddings is True
            if len(calls) == 1:
                entered.set()
                assert release.wait(TIMEOUT)
            return Vector()

    llm._embedding_model = Model()

    def run():
        try:
            results.append(llm.useLocalEmbedding("probe"))
        except Exception as exc:
            errors.append(exc)

    first = threading.Thread(target=run)
    second = threading.Thread(target=run)
    first.start()
    try:
        assert entered.wait(TIMEOUT)
        second.start()
        assert lock.second_attempt.wait(TIMEOUT)
        assert calls == ["probe"]
        assert not results
    finally:
        release.set()
        first.join(TIMEOUT)
        if second.ident is not None:
            second.join(TIMEOUT)
    assert not first.is_alive() and not second.is_alive()
    assert not errors
    assert results == [[1.0, 0.0], [1.0, 0.0]]


def test_model_error_does_not_block_next_encode(llm):
    class Model:
        failed = False

        def encode(self, *args, **kwargs):
            if not self.failed:
                self.failed = True
                raise ValueError("model failed")
            return types.SimpleNamespace(tolist=lambda: [1.0, 0.0])

    llm._embedding_model = Model()
    with pytest.raises(ValueError, match="model failed"):
        llm.useLocalEmbedding("probe")
    results = []
    caller = threading.Thread(
        target=lambda: results.append(llm.useLocalEmbedding("probe")), daemon=True
    )
    caller.start()
    caller.join(TIMEOUT)
    assert not caller.is_alive()
    assert results == [[1.0, 0.0]]
