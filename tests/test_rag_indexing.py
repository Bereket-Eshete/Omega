"""Startup indexing tests with event-controlled embeddings."""

import importlib.util
import logging
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEOUT = 5


class MemoryCollection:
    def __init__(self):
        self.records = {}
        self.lock = threading.Lock()

    def get(self, ids=None, where=None, include=None):
        with self.lock:
            selected = {
                key: value
                for key, value in self.records.items()
                if (ids is None or key in ids)
                and (
                    where is None
                    or all(value["metadata"].get(k) == v for k, v in where.items())
                )
            }
            return {
                "ids": list(selected),
                "metadatas": [v["metadata"] for v in selected.values()],
            }

    def delete(self, ids):
        with self.lock:
            for key in ids:
                self.records.pop(key, None)

    def upsert(self, *, ids, embeddings, documents, metadatas):
        with self.lock:
            for key, vector, text, metadata in zip(
                ids, embeddings, documents, metadatas
            ):
                self.records[key] = {
                    "embedding": vector,
                    "document": text,
                    "metadata": metadata,
                }


@pytest.fixture
def rag_fixture(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    collection = MemoryCollection()
    settings = {}
    bridge = types.ModuleType("lib_chromadb")
    bridge.COLLECTION = collection
    llm = types.ModuleType("lib_llm_ext")
    llm.initLocalEmbedding = lambda: None
    llm.useLocalEmbedding = lambda text: [1.0, 0.0, 0.0]
    monkeypatch.setitem(sys.modules, "lib_chromadb", bridge)
    monkeypatch.setitem(sys.modules, "lib_llm_ext", llm)
    monkeypatch.setitem(sys.modules, "openai", types.ModuleType("openai"))
    spec = importlib.util.spec_from_file_location(
        "rag_indexing_test", REPO_ROOT / "src/rag.py"
    )
    rag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rag)
    monkeypatch.setattr(
        rag, "config_get_by_key", lambda key, default=None: settings.get(key, default)
    )
    monkeypatch.setattr(rag, "_resolve_knowledge_dir", lambda: str(tmp_path))
    yield rag, collection, settings
    if rag._knowledge_worker is not None:
        finish(rag)


def finish(rag):
    rag._knowledge_worker.join(TIMEOUT)
    assert not rag._knowledge_worker.is_alive()


@pytest.mark.parametrize("value", [None, False, "false"])
def test_missing_or_disabled_setting_waits_for_indexing(
    rag_fixture, monkeypatch, tmp_path, value
):
    rag, collection, settings = rag_fixture
    if value is not None:
        settings["knowledgeIndexingBackground"] = value
    (tmp_path / "guide.md").write_text("Knowledge", encoding="utf-8")
    entered, release, returned = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    embedding_threads = []

    def embed(texts):
        embedding_threads.append(threading.get_ident())
        entered.set()
        assert release.wait(TIMEOUT)
        return [[1.0, 0.0, 0.0] for text in texts]

    monkeypatch.setattr(rag, "local_embed_batch", embed)
    results = []

    def startup():
        results.append(rag.init_knowledge("Local"))
        returned.set()

    caller = threading.Thread(target=startup)
    caller.start()
    try:
        assert entered.wait(TIMEOUT)
        assert embedding_threads == [caller.ident]
        assert not returned.is_set()
    finally:
        release.set()
        caller.join(TIMEOUT)
    assert not caller.is_alive()
    assert returned.is_set()
    assert results == ["Knowledge: 1 files (0 unchanged, 1 re-indexed)"]
    assert collection.records["guide.md_chunk_0"]["document"] == "Knowledge"


@pytest.mark.parametrize("value", [True, "true"])
def test_background_returns_before_completion_and_stores_records(
    rag_fixture, monkeypatch, tmp_path, value
):
    rag, collection, settings = rag_fixture
    settings["knowledgeIndexingBackground"] = value
    guide = tmp_path / "guide.md"
    guide.write_text("Knowledge", encoding="utf-8")
    entered, release = threading.Event(), threading.Event()

    def embed(texts):
        entered.set()
        assert release.wait(TIMEOUT)
        return [[1.0, 0.0, 0.0] for text in texts]

    monkeypatch.setattr(rag, "local_embed_batch", embed)
    try:
        assert rag.init_knowledge("Local") == "Knowledge indexing started in background"
        assert entered.wait(TIMEOUT)
        assert rag._knowledge_worker.daemon
        assert not collection.records
    finally:
        release.set()
    finish(rag)
    assert collection.records["guide.md_chunk_0"] == {
        "embedding": [1.0, 0.0, 0.0],
        "document": "Knowledge",
        "metadata": {
            "source": "guide.md",
            "breadcrumb": "guide.md",
            "type": "chunk",
            "time": "knowledge_prior",
        },
    }
    assert collection.records["hash_guide.md"]["metadata"]["hash"] == rag._file_hash(
        guide
    )


def test_concurrent_initializations_do_not_start_duplicate_workers(
    rag_fixture, monkeypatch, tmp_path
):
    rag, _, settings = rag_fixture
    settings["knowledgeIndexingBackground"] = True
    (tmp_path / "guide.md").write_text("Knowledge", encoding="utf-8")
    entered, release = threading.Event(), threading.Event()
    calls = []

    def embed(texts):
        calls.append(texts)
        entered.set()
        assert release.wait(TIMEOUT)
        return [[1.0, 0.0, 0.0] for text in texts]

    monkeypatch.setattr(rag, "local_embed_batch", embed)
    barrier = threading.Barrier(4)

    def initialize(_):
        barrier.wait(TIMEOUT)
        return rag.init_knowledge("Local")

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(initialize, range(4)))
        assert entered.wait(TIMEOUT)
        assert results.count("Knowledge indexing started in background") == 1
        assert results.count("Knowledge indexing already running") == 3
        settings["knowledgeIndexingBackground"] = False
        assert rag.init_knowledge("Local") == "Knowledge indexing already running"
        assert len(calls) == 1
    finally:
        release.set()
    finish(rag)


def test_background_failure_logs_traceback_and_does_not_block_later_calls(
    rag_fixture, monkeypatch, tmp_path, caplog
):
    rag, collection, settings = rag_fixture
    settings["knowledgeIndexingBackground"] = True
    (tmp_path / "guide.md").write_text("Knowledge", encoding="utf-8")

    def broken_embed(texts):
        raise ValueError("provider unavailable")

    with monkeypatch.context() as patch, caplog.at_level(logging.ERROR):
        patch.setattr(rag, "local_embed_batch", broken_embed)
        rag.init_knowledge("Local")
        finish(rag)
    assert any(
        record.exc_info and "provider unavailable" in record.message
        for record in caplog.records
    )
    assert rag.init_knowledge("Local") == "Knowledge indexing started in background"
    finish(rag)
    assert collection.records["guide.md_chunk_0"]["document"] == "Knowledge"


def test_thread_start_failure_is_logged(rag_fixture, monkeypatch, tmp_path, caplog):
    rag, collection, settings = rag_fixture
    settings["knowledgeIndexingBackground"] = True
    (tmp_path / "guide.md").write_text("Knowledge", encoding="utf-8")

    class BrokenThread:
        def __init__(self, **kwargs):
            pass

        def start(self):
            raise RuntimeError("cannot start thread")

    with monkeypatch.context() as patch, caplog.at_level(logging.ERROR):
        patch.setattr(rag.threading, "Thread", BrokenThread)
        assert "cannot start thread" in rag.init_knowledge("Local")
    assert any(record.exc_info for record in caplog.records)
    assert rag.init_knowledge("Local") == "Knowledge indexing started in background"
    finish(rag)
    assert "guide.md_chunk_0" in collection.records


@pytest.mark.parametrize("source", ["yaml", "environment", "cli"])
def test_option_uses_existing_configuration_sources(
    rag_fixture, monkeypatch, tmp_path, source
):
    import config

    rag, _, _ = rag_fixture
    for name in ("_CONFIG", "_COMMAND_LINE", "_CONFIG_FILE"):
        monkeypatch.setattr(config, name, {})
    monkeypatch.delenv("OMEGA_knowledgeIndexingBackground", raising=False)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "knowledgeIndexingBackground: " + ("true" if source == "yaml" else "false"),
        encoding="utf-8",
    )
    args = [f"config={config_file}"]
    if source == "environment":
        monkeypatch.setenv("OMEGA_knowledgeIndexingBackground", "true")
    elif source == "cli":
        args.append("knowledgeIndexingBackground=true")
    config.init_config(args)
    monkeypatch.setattr(rag, "config_get_by_key", config.config_get_by_key)
    assert rag.init_knowledge("Local") == "Knowledge indexing started in background"
    finish(rag)


def test_cloud_embedding_remains_available_while_indexing(
    rag_fixture, monkeypatch, tmp_path
):
    rag, _, settings = rag_fixture
    settings.update(knowledgeIndexingBackground=True, embeddingprovider="OpenAI")
    (tmp_path / "guide.md").write_text("Knowledge", encoding="utf-8")
    entered, release = threading.Event(), threading.Event()

    class FakeClient:
        def __init__(self):
            self.embeddings = self

        def create(self, *, model, input):
            if input == ["Knowledge"]:
                entered.set()
                assert release.wait(TIMEOUT)
            return types.SimpleNamespace(
                data=[
                    types.SimpleNamespace(embedding=[1.0, 0.0, 0.0]) for text in input
                ]
            )

    monkeypatch.setattr(rag.openai, "OpenAI", FakeClient, raising=False)
    try:
        rag.init_knowledge("OpenAI")
        assert entered.wait(TIMEOUT)
        assert rag.cloud_embed("runtime probe") == [1.0, 0.0, 0.0]
    finally:
        release.set()
    finish(rag)
