import os
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "omega"
CONTAINER_TEST_IMAGE = os.environ.get("OMEGA_LAUNCHER_TEST_IMAGE", "")
UPLOAD_PREFIX_SIZE = 1024


def _load_installer_namespace():
    launcher_source = LAUNCHER.read_text(encoding="utf-8")
    installer_source = launcher_source.split("cat >\"$tmp_py_file\" <<'PY'\n", 1)[1].split(
        "\nPY\n", 1
    )[0]
    namespace = {"__name__": "omega_installer"}
    exec(compile(installer_source, str(LAUNCHER), "exec"), namespace)
    return namespace


def _stub_docker_environment(tmp_path: Path) -> dict:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        "printf 'docker'\n"
        "printf ' <%s>' \"$@\"\n"
        "printf '\\n'\n",
        encoding="utf-8",
    )

    docker.chmod(0o755)
    environment = os.environ.copy()
    environment["ASI_API_KEY"] = "test-token"
    environment["PATH"] = (
        f"{bin_dir}{os.pathsep}{Path(sys.executable).parent}"
        f"{os.pathsep}{environment['PATH']}"
    )
    return environment


def _run_launcher(
    tmp_path: Path, *options: str, memory_import: bool = True
) -> subprocess.CompletedProcess:
    import_options = ["--memory-import", "memory.tar.gz"] if memory_import else []
    return subprocess.run(
        [str(LAUNCHER), "start", *import_options, *options],
        cwd=REPO_ROOT,
        env=_stub_docker_environment(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )


def _memory_upload_command(directory: Path, size: int) -> list[str]:
    source = LAUNCHER.read_text(encoding="utf-8").split("memory_upload() {", 1)[1]
    script = source.split("memory_transfer_run rw -c '\n", 1)[1].split("\n' ", 1)[0]
    return [
        sys.executable, "-B", "-c", script.replace("/memory-transfer", str(directory)),
        "memory.tar.gz", str(size),
    ]


@pytest.fixture
def pending_upload(tmp_path):
    content = b"(memory archive)\x00\xff\n" * 8192
    command = _memory_upload_command(tmp_path, len(content))
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        process.stdin.write(content[:UPLOAD_PREFIX_SIZE])
        process.stdin.flush()
        deadline = time.monotonic() + 5
        while not any(tmp_path.rglob("*.tar.gz")):
            assert process.poll() is None
            if time.monotonic() >= deadline:
                pytest.fail("upload did not start")
            time.sleep(0.01)
        assert not (tmp_path / "memory.tar.gz").exists()
        yield process, command, content
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_installer_preserves_volume_and_explicit_export_setting(tmp_path, monkeypatch, enabled):
    installer = _load_installer_namespace()
    monkeypatch.setitem(installer, "require_license_acceptance", lambda: None)
    monkeypatch.setitem(installer, "_choose_channel", lambda: {"commchannel": "test"})
    monkeypatch.setitem(
        installer, "_choose_provider", lambda: ("Test", "Local", "TEST_SERVER_IP", "", "", "localhost")
    )
    monkeypatch.setitem(installer, "_choose_import_kb", lambda: "0")
    monkeypatch.setattr(
        "builtins.input", lambda _: "n" if enabled == "0" else pytest.fail("unexpected prompt")
    )
    output = tmp_path / "config"

    installer["config_run_omega"](str(output), "test-archives", enabled)

    config = output.read_text(encoding="utf-8")
    assert "memory_transfer_volume_name=test-archives\n" in config
    assert f"memory_export_enabled={enabled}\n" in config


def test_launcher_mounts_named_transfer_volume_for_nonroot_agent(tmp_path):
    result = _run_launcher(
        tmp_path, "--memory-transfer-volume", "test-archives", "--enable-memory-export", memory_import=False
    )

    assert result.returncode == 0, result.stderr
    agent_run = next(
        line for line in result.stdout.splitlines()
        if line.startswith("docker <run> ") and "<--name> <omega>" in line
    )
    assert "<type=volume,source=test-archives,target=/memory-transfer>" in agent_run
    assert "<--user> <65534:65534>" in result.stdout
    assert "<memoryExportEnabled=true>" in agent_run
    assert "MEMORY_TRANSFER_GID" not in result.stdout


@pytest.mark.parametrize("volume", ["/tmp/archives", "../archives", "archives:/host", "archives,readonly"])
def test_launcher_rejects_host_paths_as_volume_names(tmp_path, volume):
    result = _run_launcher(tmp_path, "--memory-transfer-volume", volume, memory_import=False)

    assert result.returncode != 0
    assert "must be a Docker volume name" in result.stderr
    assert "docker <" not in result.stdout


@pytest.mark.parametrize("volume", ["omega-memory", "omegaclaw-memory"])
def test_launcher_rejects_live_memory_volume_names(tmp_path, volume):
    result = _run_launcher(tmp_path, "--memory-transfer-volume", volume, memory_import=False)

    assert result.returncode != 0
    assert "must not use a live memory volume" in result.stderr
    assert "docker <" not in result.stdout


@pytest.mark.parametrize("option", ["--memory-transfer-dir", "--memory-transfer-gid"])
def test_launcher_explains_migration_from_bind_mount_options(tmp_path, option):
    result = _run_launcher(tmp_path, option, "old-value", memory_import=False)

    assert result.returncode != 0
    assert "has been replaced by --memory-transfer-volume" in result.stderr
    assert "docker <" not in result.stdout


def test_clean_preserves_memory_archives(tmp_path):
    result = subprocess.run(
        [str(LAUNCHER), "clean"], env=_stub_docker_environment(tmp_path), capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert "docker <volume> <rm> <omega-memory>" in result.stdout
    assert "omega-memory-transfer" not in result.stdout


def test_download_rejects_archive_path_traversal(tmp_path):
    result = subprocess.run(
        [str(LAUNCHER), "memory-download", "../memory.tar.gz", str(tmp_path / "download")],
        env=_stub_docker_environment(tmp_path), capture_output=True, text=True,
    )

    assert result.returncode != 0
    assert "must be a plain .tar.gz filename" in result.stderr
    assert "docker <" not in result.stdout


@pytest.mark.parametrize("command", ["memory-upload", "memory-download"])
def test_archive_commands_require_arguments_before_docker(tmp_path, command):
    result = subprocess.run(
        [str(LAUNCHER), command], env=_stub_docker_environment(tmp_path),
        capture_output=True, text=True,
    )

    assert result.returncode != 0
    assert f"{command} requires" in result.stderr
    assert "docker <" not in result.stdout


def test_download_preserves_existing_destination(tmp_path):
    destination = tmp_path / "memory.tar.gz"
    destination.write_bytes(b"existing archive")
    result = subprocess.run(
        [str(LAUNCHER), "memory-download", "memory.tar.gz", str(destination)],
        env=_stub_docker_environment(tmp_path), capture_output=True, text=True,
    )

    assert result.returncode != 0
    assert "destination already exists" in result.stderr
    assert destination.read_bytes() == b"existing archive"


def test_failed_download_removes_partial_destination(tmp_path):
    environment = _stub_docker_environment(tmp_path)
    (tmp_path / "bin" / "docker").write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        "  volume) exit 0 ;;\n"
        "  run) printf partial; exit 1 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    destination = tmp_path / "download.tar.gz"
    result = subprocess.run(
        [str(LAUNCHER), "memory-download", "memory.tar.gz", str(destination)],
        env=environment, capture_output=True, text=True,
    )

    assert result.returncode != 0
    assert not destination.exists()


def test_launcher_warns_before_overwrite_memory_import(tmp_path):
    result = _run_launcher(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "WARNING: --memory-mode overwrite will replace the selected current memory components" in result.stderr


def test_launcher_does_not_warn_for_append_memory_import(tmp_path):
    result = _run_launcher(tmp_path, "--memory-mode", "append")

    assert result.returncode == 0, result.stderr
    assert "WARNING:" not in result.stderr


def test_upload_publishes_only_completed_archive(pending_upload, tmp_path):
    process, _, content = pending_upload
    _, error = process.communicate(input=content[UPLOAD_PREFIX_SIZE:], timeout=5)

    assert process.returncode == 0, error.decode()
    assert (tmp_path / "memory.tar.gz").read_bytes() == content
    assert [path.name for path in tmp_path.iterdir()] == ["memory.tar.gz"]


def test_incomplete_upload_is_cleaned_up(pending_upload, tmp_path):
    process, _, _ = pending_upload
    _, error = process.communicate(timeout=5)

    assert process.returncode != 0
    assert b"Archive upload was incomplete" in error
    assert not list(tmp_path.iterdir())


def test_interrupted_upload_can_be_retried(pending_upload, tmp_path):
    process, command, content = pending_upload
    process.kill()
    process.communicate(timeout=5)
    assert not (tmp_path / "memory.tar.gz").exists()

    result = subprocess.run(command, input=content, capture_output=True, timeout=5)

    assert result.returncode == 0, result.stderr.decode()
    assert (tmp_path / "memory.tar.gz").read_bytes() == content


def test_upload_preserves_competing_archive(pending_upload, tmp_path):
    process, command, content = pending_upload
    competing = b"B" * len(content)
    result = subprocess.run(command, input=competing, capture_output=True, timeout=5)
    assert result.returncode == 0, result.stderr.decode()

    _, error = process.communicate(input=content[UPLOAD_PREFIX_SIZE:], timeout=5)

    assert process.returncode != 0
    assert b"Archive already exists" in error
    assert (tmp_path / "memory.tar.gz").read_bytes() == competing
    assert [path.name for path in tmp_path.iterdir()] == ["memory.tar.gz"]


@pytest.mark.skipif(
    not CONTAINER_TEST_IMAGE or shutil.which("docker") is None,
    reason="needs Docker and OMEGA_LAUNCHER_TEST_IMAGE",
)
def test_memory_volume_upload_download_round_trip(tmp_path):
    volume = "omega-transfer-test-" + uuid.uuid4().hex
    directory = tmp_path / "archives with spaces"
    directory.mkdir()
    content = directory / "history.metta"
    content.write_bytes(b"(sample history)\n\x00\xff\r\n")
    source = directory / "memory.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        archive.add(content, arcname="history.metta")
    original = source.read_bytes()

    def run(command, *arguments):
        return subprocess.run(
            [str(LAUNCHER), command, *map(str, arguments), "-d", CONTAINER_TEST_IMAGE,
             "--memory-transfer-volume", volume],
            capture_output=True, text=True, check=False,
        )

    try:
        uploaded = run("memory-upload", source)
        assert uploaded.returncode == 0, uploaded.stderr
        listed = run("memory-list")
        assert listed.returncode == 0, listed.stderr
        assert listed.stdout.strip() == source.name

        permissions = subprocess.run(
            ["docker", "run", "--rm", "--pull=never", "--user", "65534:65534",
             "--mount", f"type=volume,source={volume},target=/memory-transfer,readonly",
             "--entrypoint", "/bin/sh", CONTAINER_TEST_IMAGE, "-c",
             "stat -c '%u:%g %a' /memory-transfer /memory-transfer/memory.tar.gz"],
            capture_output=True, text=True, check=False,
        )
        assert permissions.returncode == 0, permissions.stderr
        assert permissions.stdout.splitlines() == ["65534:65534 700", "65534:65534 600"]

        source.write_bytes(b"replacement must not overwrite the archive")
        duplicate = run("memory-upload", source)
        assert duplicate.returncode != 0
        assert "Archive already exists" in duplicate.stderr

        destination = directory / "downloaded archive.tar.gz"
        downloaded = run("memory-download", source.name, destination)
        assert downloaded.returncode == 0, downloaded.stderr
        assert destination.read_bytes() == original
        assert destination.stat().st_mode & 0o777 == 0o600
        existing = run("memory-download", source.name, destination)
        assert existing.returncode != 0
        assert destination.read_bytes() == original

        missing = directory / "missing archive.tar.gz"
        failed = run("memory-download", "missing.tar.gz", missing)
        assert failed.returncode != 0
        assert not missing.exists()
    finally:
        subprocess.run(["docker", "volume", "rm", volume], capture_output=True, check=False)


@pytest.mark.parametrize(
    ("option", "included_environment", "excluded_environment"),
    [
        ("--only-history", "MEMORY_IMPORT_NO_VECTOR=1", "MEMORY_IMPORT_NO_HISTORY=1"),
        ("--only-vector", "MEMORY_IMPORT_NO_HISTORY=1", "MEMORY_IMPORT_NO_VECTOR=1"),
    ],
)
def test_only_component_options_select_one_import_component(
    tmp_path,
    option,
    included_environment,
    excluded_environment,
):
    result = _run_launcher(tmp_path, option)

    assert result.returncode == 0, result.stderr
    assert included_environment in result.stdout
    assert excluded_environment not in result.stdout


def test_only_component_options_are_mutually_exclusive(tmp_path):
    result = _run_launcher(tmp_path, "--only-history", "--only-vector")

    assert result.returncode != 0
    assert "--only-history and --only-vector cannot be combined" in result.stderr


@pytest.mark.parametrize("command", ["start", "stop", "clean", "memory-download"])
@pytest.mark.parametrize(
    ("flag", "expected_output"),
    [
        ("-h", "Usage:"),
        ("--help", "Usage:"),
        ("-v", "Omega version="),
        ("--version", "Omega version="),
    ],
)
def test_help_and_version_flags_do_not_run_the_command(tmp_path, command, flag, expected_output):
    result = subprocess.run(
        [str(LAUNCHER), command, flag],
        cwd=REPO_ROOT,
        env=_stub_docker_environment(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert expected_output in result.stdout
    assert "docker <" not in result.stdout


@pytest.mark.parametrize("removed_option", ["--no-history", "--no-vector"])
def test_removed_component_options_are_rejected(tmp_path, removed_option):
    result = _run_launcher(tmp_path, removed_option)

    assert result.returncode != 0
    assert "Usage:" in result.stdout
    assert "docker <" not in result.stdout
