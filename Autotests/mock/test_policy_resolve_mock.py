"""
Mock test for the security policy path resolution.

profile/policy.py resolves the policy paths before Landlock sees them: a
glob pattern is expanded to the paths present on the host machine, and a
path matching nothing is dropped, because Landlock cannot add a rule for
a path which does not exist. The resolution is checked inside the
container, the only place where py_landlock is installed.

pytest test_policy_resolve_mock.py -s
"""
import json

from helpers import Checker, dexec

NODES_DIR = "/tmp/policy_nodes"
POLICY_PATHS = [f"{NODES_DIR}/node[0-9]", f"{NODES_DIR}/absent*",
                f"{NODES_DIR}/absent"]
EXPECTED_PATHS = [f"{NODES_DIR}/node0", f"{NODES_DIR}/node1"]
PROBE = (
    "import json, sys;"
    "sys.path[:0] = ['/PeTTa/repos/Omega/profile', '/PeTTa/repos/Omega',"
    " '/PeTTa/repos/Omega/src'];"
    "import policy;"
    "p = policy.FileSystemPolicy();"
    "p.load_dict({'version': 1, 'filesystem_policy': {'read_write': "
    f"{POLICY_PATHS}"
    "}});"
    "print(json.dumps([str(path) for path in p._resolve_paths(p._read_write)]))"
)


def test_policy_resolves_paths():
    with Checker("policy path resolution mock", cleanup_dirs=[NODES_DIR]) as c:
        c.step("create the paths the policy pattern must match")
        result = dexec("mkdir", "-p", NODES_DIR)
        if result.returncode != 0:
            c.fail("nodes", f"could not create {NODES_DIR}")
        result = dexec("touch", f"{NODES_DIR}/node0", f"{NODES_DIR}/node1")
        if result.returncode != 0:
            c.fail("nodes", f"could not create files in {NODES_DIR}")
        c.ok("nodes", f"{NODES_DIR} contains node0 and node1")

        c.step("resolve the policy paths inside the container")
        result = dexec("python3", "-c", PROBE)
        if result.returncode != 0:
            c.fail("probe", f"policy module failed to resolve paths: {result.stderr}")
        try:
            resolved = json.loads(result.stdout.strip())
        except json.JSONDecodeError as e:
            c.fail("probe", f"could not parse the resolved paths: {e}\nOutput was: {result.stdout}")
        c.ok("probe", f"resolved {resolved}")

        c.step("compare the resolved paths with the expected ones")
        if sorted(resolved) != sorted(EXPECTED_PATHS):
            c.fail(
                name="resolve",
                detail=(
                    f"Path resolution mismatch! The policy paths {POLICY_PATHS} "
                    f"resolved to {sorted(resolved)}, expected {sorted(EXPECTED_PATHS)}."
                )
            )
        c.ok("resolve", "the pattern is expanded and paths matching nothing are dropped")
        c.done()
