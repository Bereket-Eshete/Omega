"""Real persistent Chroma + PeTTa bridge; put the bridge directory on PYTHONPATH."""

import importlib.util
import os
import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEOUT = 30


def test_background_indexing_and_runtime_memory_share_persistent_collection(
    monkeypatch, tmp_path
):
    pytest.importorskip("chromadb", reason="real Chroma integration requires chromadb")
    bridge_spec = importlib.util.find_spec("lib_chromadb")
    if bridge_spec is None:
        pytest.skip("PeTTa lib_chromadb bridge is required on PYTHONPATH")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    # Execute the actual bridge in an isolated directory, including its client creation.
    spec = importlib.util.spec_from_file_location("lib_chromadb", bridge_spec.origin)
    bridge = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "lib_chromadb", bridge)
    spec.loader.exec_module(bridge)
    llm = types.ModuleType("lib_llm_ext")
    llm.initLocalEmbedding = lambda: None
    llm.useLocalEmbedding = lambda text: [1.0, 0.0, 0.0]
    monkeypatch.setitem(sys.modules, "lib_llm_ext", llm)
    spec = importlib.util.spec_from_file_location(
        "rag_chroma_test", REPO_ROOT / "src/rag.py"
    )
    rag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rag)
    monkeypatch.setattr(
        rag,
        "config_get_by_key",
        lambda key, default=None: (
            True if key == "knowledgeIndexingBackground" else default
        ),
    )
    knowledge = tmp_path / "knowledge-priors"
    knowledge.mkdir()
    # Enough records to exercise the HNSW batch buffer as well as its index.
    for i in range(120):
        (knowledge / f"{i:03}.md").write_text(f"prior {i:03}", encoding="utf-8")
    monkeypatch.setattr(rag, "_resolve_knowledge_dir", lambda: str(knowledge))
    entered, release = threading.Event(), threading.Event()

    def embed(texts):
        if texts == ["prior 060"]:
            entered.set()
            assert release.wait(TIMEOUT)
        return [[1.0, 0.0, 0.0] for text in texts]

    monkeypatch.setattr(rag, "local_embed_batch", embed)
    try:
        assert rag._get_collection() is bridge.COLLECTION
        assert rag.init_knowledge("Local") == "Knowledge indexing started in background"
        assert entered.wait(TIMEOUT)
        assert rag._knowledge_worker.is_alive()
        assert bridge.COLLECTION.count() == 120  # 60 files, each with a hash sentinel.
        assert not bridge.COLLECTION.get(ids=["060.md_chunk_0"])["ids"]
        results = bridge.query([1.0, 0.0, 0.0], 5)
        assert all(
            time == "knowledge_prior" and text.startswith("prior ")
            for time, text in results
        )
        bridge.remember("runtime while indexing", [0.0, 1.0, 0.0], "runtime")
        assert bridge.query([0.0, 1.0, 0.0], 1) == [
            ["runtime", "runtime while indexing"]
        ]
    finally:
        release.set()
    try:
        # Foreground reads/writes continue as the remaining background upserts run.
        for i in range(25):
            bridge.remember(f"runtime {i}", [0.0, 1.0, 0.0], "runtime")
            assert bridge.query([1.0, 0.0, 0.0], 1)[0][0] == "knowledge_prior"
    finally:
        rag._knowledge_worker.join(TIMEOUT)
    assert not rag._knowledge_worker.is_alive()
    assert bridge.COLLECTION.count() == 266
    assert bridge.COLLECTION.get(ids=["119.md_chunk_0"])["documents"] == ["prior 119"]

    # A separate process reopens persistence and must skip every completed file.
    script = """
import sys, types
from pathlib import Path
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, str(Path(sys.argv[1]) / "src"))
llm = types.ModuleType("lib_llm_ext")
def unexpected_embedding(*args):
    raise AssertionError("unchanged file was embedded again")
llm.initLocalEmbedding = unexpected_embedding
llm.useLocalEmbedding = unexpected_embedding
sys.modules["lib_llm_ext"] = llm
import lib_chromadb, rag
rag.config_get_by_key = lambda key, default=None: default
rag._resolve_knowledge_dir = lambda: sys.argv[2]
assert lib_chromadb.COLLECTION.count() == 266
assert lib_chromadb.query([0.0, 1.0, 0.0], 1)[0][0] == "runtime"
assert rag.init_knowledge("Local") == "Knowledge: 120 files (120 unchanged, 0 re-indexed)"
print("Persistent restart: 120 unchanged files skipped; runtime memory retrievable")
"""
    env = dict(os.environ, PYTHONPATH=str(Path(bridge_spec.origin).parent))
    result = subprocess.run(
        [sys.executable, "-c", script, str(REPO_ROOT), str(knowledge)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "120 unchanged files skipped" in result.stdout
