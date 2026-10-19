import json
import hashlib
from typing import Dict, Iterator, List, Optional, Tuple
import os
import re
import subprocess
from collections import deque
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    from src.logger import get_logger
except ModuleNotFoundError:  # running this file directly as a script
    from logger import get_logger

logger = get_logger(__name__)

TS_RE = re.compile(r'^\("(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"')
# Consume quoted strings as whole tokens, including an unfinished final string,
# so parentheses in message text never become structural delimiters.
_FORM_TOKENS_RE = re.compile(r'"(?:\\.|[^"\\])*(?:"|\\?\Z)|[()]', re.DOTALL)
_FIELD_HEAD_RE = re.compile(r'\(([^\s()"]+)(?=\s|\()\s*')
_FIELD_VALUE_RE = re.compile(r'"(?:\\.|[^"\\])*"|[^\s()"][^\s)]*', re.DOTALL)
LLM_COMMANDS = {
    "append-file",
    "clear-frame-junk",
    "compact-frame",
    "complete-goals-ltm",
    "complete-goals-stm",
    "ctx-add-hypothesis",
    "ctx-add-result",
    "episodes",
    "metta",
    "new-autonomous-frame",
    "new-frame",
    "pin",
    "query",
    "read-file",
    "remember",
    "websearch",
    "send",
    "send_probe",
    "shell",
    "show-active-framespace",
    "show-completed-framespace",
    "show-current-frame",
    "show-frame-index",
    "show-frame-relation",
    "show-root-frame",
    "switch-frame",
    "switch-mode",
    "tavily-search",
    "technical-analysis",
    "write-file",
    "get-io-policy",
    "write-file-b64",
}
TWO_ARG_COMMANDS = {
    "write-file",
    "append-file",
    "write-file-b64",
    "ctx-add-hypothesis",
    "ctx-add-result",
}

def compact_plain(value, limit=1200):
    """
    Return a compact, single-line summary with a stable digest.
    This does not write files and does not store to LTM.
    MeTTa decides whether to pin/remember the resulting summary.
    """
    text = normalize_string(value)
    compact = re.sub(r"\s+", " ", text).strip()
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

    if len(compact) > int(limit):
        compact = compact[: int(limit) - 3].rstrip() + "..."

    return f"sha256:{digest[:16]} chars:{len(text)} excerpt:{compact}"


def _unquote_history_string(value):
    """Remove S-expression quotes and only decode escaped quotes/backslashes."""
    text = str(value)
    if len(text) < 2 or text[0] not in ('"', "'") or text[-1] != text[0]:
        return text
    inner = text[1:-1]
    out = []
    i = 0
    while i < len(inner):
        if inner[i] == "\\" and i + 1 < len(inner) and inner[i + 1] in ('"', "'", "\\"):
            out.append(inner[i + 1])
            i += 2
        else:
            out.append(inner[i])
            i += 1
    return "".join(out)


def cfv2_history_payload(value, limit=900):
    """Normalize, cap, and JSON-quote a payload for safe MeTTa event storage."""
    text = normalize_string(value).strip()
    text = _unquote_history_string(text)
    text = re.sub(r"\s+", " ", text).strip()
    limit = max(0, int(limit))
    if len(text) <= limit:
        compact = text
    elif limit <= 3:
        compact = text[:limit]
    else:
        compact = text[: limit - 3].rstrip() + "..."
    # The caller stores this value directly in a FrameEvent field. Returning a
    # JSON string literal keeps whitespace and escaped characters parseable
    # without routing the payload through MeTTa's swrite function.
    return json.dumps(compact, ensure_ascii=False)


def cfv2_compact_history(events_repr, limit=2400):
    """Format the newest frame events into a plain-text summary within `limit`."""
    limit = max(0, int(limit))
    if limit == 0:
        return ""

    events = _balanced_exprs(normalize_string(events_repr), "FrameEvent")
    lines = []
    for event in events:
        fields = _fields(event, "category", "kind", "timestamp", "payload")
        category = fields.get("category") or "Event"
        kind = fields.get("kind") or category
        timestamp = fields.get("timestamp") or ""
        payload = fields.get("payload") or ""
        timestamp = _unquote_history_string(timestamp)
        if len(payload) >= 2 and payload[0] == '"' and payload[-1] == '"':
            try:
                payload = json.loads(payload)
            except (TypeError, ValueError):
                payload = _unquote_history_string(payload)
        else:
            payload = _unquote_history_string(payload)
        payload = re.sub(r"\s+", " ", payload).strip()
        stamp = f" [{timestamp}]" if timestamp else ""
        lines.append(f"{category}/{kind}{stamp}: {payload}")

    # Preserve recent context first when space is limited, but render the result
    # in chronological order for readability.
    selected = []
    used = 0
    for line in reversed(lines):
        separator = 1 if selected else 0
        remaining = limit - used - separator
        if remaining <= 0:
            break
        if len(line) > remaining:
            # Older events should be omitted rather than included as a broken
            # fragment. If the newest event alone exceeds the limit, retain its
            # header and end the payload with an ellipsis at a word boundary.
            if selected:
                break
            header, delimiter, payload = line.partition(": ")
            marker = "..."
            payload_budget = remaining - len(header) - len(delimiter) - len(marker)
            if not delimiter or payload_budget <= 0:
                break
            excerpt = payload[:payload_budget].rstrip()
            if len(excerpt) < len(payload):
                excerpt = excerpt.rsplit(" ", 1)[0] if " " in excerpt else ""
            line = f"{header}: {excerpt}{marker}"
        selected.append(line)
        used += len(line) + separator
    selected.reverse()
    return "\n".join(selected)


def make_id(prefix="id"):
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}-{stamp}"


def extract_timestamp(line):
    m = TS_RE.search(line)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError as e:
        logger.error(f"Line does not carry a parsable timestamp: {e}")
        return None

def around_time(needle_time_str, k):
    needle_time_str = needle_time_str.replace(r'\"', '').replace('"', '').strip()
    filename = "repos/Omega/memory/history.metta"
    target = datetime.strptime(needle_time_str, "%Y-%m-%d %H:%M:%S")
    best_lineno = None
    best_line = None
    best_diff = None
    buffer = []
    best_idx = None
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            buffer.append((lineno, line))
            ts = extract_timestamp(line)
            if ts is None:
                continue
            diff = abs((ts - target).total_seconds())
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_lineno = lineno
                best_line = line
                best_idx = len(buffer) - 1
    if best_lineno is None:
        return
    start = max(0, best_idx - k)
    end = min(len(buffer), best_idx + k + 1)
    ret = ""
    for lineno, line in buffer[start:end]:
        ret += f"{lineno}:{line}"
    return ret

def quote_arg(x):
    if x.startswith('"') and x.endswith('"') and "\n" not in x:
        return x
    else:
        return json.dumps(x, ensure_ascii=False)

def starts_command_line(line):
    s = line.lstrip()
    if not s:
        return False
    # allow "(send ...)" as command start too
    if s.startswith("("):
        s = s[1:].lstrip()
    if not s:
        return False
    first = s.split(maxsplit=1)[0].rstrip(")")
    return first in LLM_COMMANDS

def split_toplevel_forms(line):
    """Split a line holding several complete s-expressions into separate forms.

    Parentheses inside string literals are ignored. A line that is not a plain
    sequence of balanced top-level forms is returned unchanged, so single-form
    answers and free text keep their previous handling.
    """
    forms = []
    depth = 0
    start = None
    in_string = False
    escaped = False
    for i, ch in enumerate(line):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start is not None:
                forms.append(line[start:i + 1])
                start = None
            elif depth < 0:
                return [line]
        elif depth == 0 and not ch.isspace():
            return [line]
    if depth != 0 or len(forms) < 2:
        return [line]
    return forms


def split_command_blocks(s):
    blocks = []
    cur = []
    for raw in s.splitlines():
        if not raw.strip():
            if cur:
                cur.append(raw)
            continue
        if starts_command_line(raw) and cur:
            blocks.append("\n".join(cur).strip())
            cur = [raw]
        else:
            cur.append(raw)
    if cur:
        blocks.append("\n".join(cur).strip())
    expanded = []
    for block in blocks:
        expanded.extend(split_toplevel_forms(block.strip()))
    return expanded

def balance_parentheses(s):
    s = s.replace("_quote_", '"').replace("_newline_", "\n")
    sexprs = []
    for line in split_command_blocks(s):
        line = line.strip()
        if not line:
            continue
        if line.startswith("(-"):
            line = "(pin " + line[2:]
        elif line.startswith("-"):
            line = "pin " + line[1:]
        # remove one outer (...) if present
        if line.startswith("(") and line.endswith(")"):
            line = line[1:-1].strip()
        elif line.startswith("("):
            line = line[1:].strip()
        parts = line.split(maxsplit=1)
        if not parts:
            continue
        cmd = parts[0]
        rest = parts[1].strip() if len(parts) > 1 else ""
        if cmd in TWO_ARG_COMMANDS:
            if not rest:
                sexprs.append(f"({cmd})")
                continue
            # filename is first token unless already quoted
            if rest.startswith('"'):
                end = 1
                escaped = False
                while end < len(rest):
                    ch = rest[end]
                    if ch == '"' and not escaped:
                        break
                    escaped = (ch == '\\' and not escaped)
                    if ch != '\\':
                        escaped = False
                    end += 1
                if end < len(rest) and rest[end] == '"':
                    filename = rest[:end+1]
                    content = rest[end+1:].strip()
                else:
                    filename = quote_arg(rest[1:])
                    content = ""
            else:
                split_rest = rest.split(maxsplit=1)
                filename = quote_arg(split_rest[0])
                content = split_rest[1].strip() if len(split_rest) > 1 else ""
            if content:
                sexprs.append(f"({cmd} {filename} {quote_arg(content)})")
            else:
                sexprs.append(f"({cmd} {filename})")
            continue
        if rest:
            sexprs.append(f"({cmd} {quote_arg(rest)})")
        else:
            sexprs.append(f"({cmd})")
    ret = " ".join(sexprs)
    return "(" + ret + ")"

def normalize_string(x):
    try:
        if isinstance(x, bytes):
            return x.decode("utf-8", errors="ignore")
        return str(x).encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    except Exception as e:
        logger.debug(f"Could not normalize value, using its plain string form: {e}")
        return str(x)

def joinPath(parts):
    return os.path.join(*parts)

def projectRootDirectory():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- HyperClaw Context Frames V2 helper additions ----

def cfv2_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _unescape_repr_id(value: str) -> str:
    value = str(value).strip()
    value = value.replace("'", "").replace('"', "")
    value = value.replace("[", "").replace("]", "")
    return value.strip()


def _form_spans(text: str, offset: int = 0) -> Iterator[Tuple[int, int, int]]:
    """Yield balanced forms as (start, end, depth), ignoring quoted contents."""
    starts = []
    depth = 0
    for token in _FORM_TOKENS_RE.finditer(text, offset):
        if token[0] == "(":
            starts.append((token.start(), depth))
            depth += 1
        elif token[0] == ")":
            depth -= 1
            if starts:
                start, form_depth = starts.pop()
                yield start, token.end(), form_depth


def _balanced_exprs(text: str, head: str) -> List[str]:
    """Extract matching forms in source order, excluding nested matches."""
    text = str(text)
    token = f"({head}"
    matches = []
    for start, end, _ in _form_spans(text):
        after_head = start + len(token)
        if (
            text.startswith(token, start)
            and after_head < end
            and (text[after_head].isspace() or text[after_head] == "(")
        ):
            matches.append((start, end))

    out = []
    previous_end = 0
    for start, end in sorted(matches):
        if start >= previous_end:
            out.append(text[start:end])
            previous_end = end
    return out


def _field_value(expr: str, start: int) -> Optional[str]:
    """Read one quoted, nested, or bare value without decoding it."""
    if expr.startswith("(", start):
        end = next((end for _, end, depth in _form_spans(expr, start) if depth == 0), None)
        return expr[start:end] if end is not None else None
    value = _FIELD_VALUE_RE.match(expr, start)
    if value is not None:
        return value[0]
    return "" if start < len(expr) and expr[start] != '"' else None


def _fields(expr: str, *field_names: str) -> Dict[str, str]:
    """Read requested direct-child fields once, keeping each name's first value."""
    expr = str(expr)
    fields = {}
    remaining = set(field_names)
    depth = 0
    for token in _FORM_TOKENS_RE.finditer(expr):
        if token[0] == ")":
            depth -= 1
            continue
        if token[0] != "(":
            continue
        depth += 1
        if depth != 2:
            continue
        head = _FIELD_HEAD_RE.match(expr, token.start())
        if head is None or head[1] not in remaining:
            continue
        name = head[1]
        value = _field_value(expr, head.end())
        if value is not None:
            fields[name] = value
        remaining.remove(name)
        if not remaining:
            break
    return fields


def _field(expr: str, field_name: str) -> Optional[str]:
    """Return the raw value of a direct child `(field value)` form."""
    return _fields(expr, field_name).get(field_name)


def cfv2_refs_completed_after(index_repr, date_prefix) -> str:
    """Return completed FrameRefs whose completed-timestamp starts with or compares after date_prefix.

    date_prefix can be YYYY-MM-DD or a longer timestamp prefix. This is intentionally simple.
    """
    prefix = _unescape_repr_id(date_prefix)
    refs = []
    for ref in _balanced_exprs(str(index_repr), "FrameRef"):
        fields = _fields(ref, "status", "completed-timestamp")
        status = _unescape_repr_id(fields.get("status") or "")
        t = _unescape_repr_id(fields.get("completed-timestamp") or "")
        if status == "Completed" and t and t >= prefix:
            refs.append(ref)
    return "(" + " ".join(refs) + ")"


def cfv2_select_next_frame_id(index_repr, root_mode="Fast") -> str:
    """Select highest-priority active frame matching root mode from FrameRef space.

    If multiple FrameRefs exist for a frame, the last one wins. This supports append-only refs.
    """
    mode = _unescape_repr_id(root_mode)
    latest: Dict[str, Tuple[float, str, str, str]] = {}
    for ref in _balanced_exprs(str(index_repr), "FrameRef"):
        fields = _fields(ref, "frameID", "status", "frame-mode", "space", "priority")
        fid = _unescape_repr_id(fields.get("frameID") or "")
        status = _unescape_repr_id(fields.get("status") or "")
        frame_mode = _unescape_repr_id(fields.get("frame-mode") or "")
        space = _unescape_repr_id(fields.get("space") or "")
        priority_raw = _unescape_repr_id(fields.get("priority") or "0")
        try:
            priority = float(priority_raw)
        except Exception:
            priority = 0.0
        if fid:
            latest[fid] = (priority, status, frame_mode, space)

    best_id = "NON"
    best_priority = float("-inf")
    for fid, (priority, status, frame_mode, space) in latest.items():
        if space == "Active" and status in {"Active", "Focused"} and frame_mode == mode:
            if priority > best_priority:
                best_priority = priority
                best_id = fid
    return best_id

def test_balance_parenthesis():
    assert balance_parentheses('(write-file test.txt hello world)') == '((write-file "test.txt" "hello world"))'
    assert balance_parentheses('(append-file test.txt hello world)') == '((append-file "test.txt" "hello world"))'
    assert balance_parentheses('(write-file-b64 test.txt aGVsbG8=)') == '((write-file-b64 "test.txt" "aGVsbG8="))'
    assert balance_parentheses('write-file-b64 test.txt aGVsbG8=') == '((write-file-b64 "test.txt" "aGVsbG8="))'
    assert balance_parentheses('(write-file "test.txt" hello world)') == '((write-file "test.txt" "hello world"))'
    assert balance_parentheses('(write-file "test.txt" "hello world")') == '((write-file "test.txt" "hello world"))'
    assert balance_parentheses('(write-file test.txt "hello world")') == '((write-file "test.txt" "hello world"))'
    assert balance_parentheses('(send test.xt hello world)') == '((send "test.xt hello world"))'
    assert balance_parentheses('write-file test.txt hello world') == '((write-file "test.txt" "hello world"))'
    assert balance_parentheses('append-file test.txt hello world') == '((append-file "test.txt" "hello world"))'
    assert balance_parentheses('write-file "test.txt" hello world') == '((write-file "test.txt" "hello world"))'
    assert balance_parentheses('write-file "test.txt" "hello world"') == '((write-file "test.txt" "hello world"))'
    assert balance_parentheses('write-file test.txt "hello world"') == '((write-file "test.txt" "hello world"))'
    assert balance_parentheses('send test.xt hello world') == '((send "test.xt hello world"))'
    assert balance_parentheses('send Here are the planets:\n1. Mercury\n2. Venus') == '((send "Here are the planets:\\n1. Mercury\\n2. Venus"))'
    assert balance_parentheses('send Here are the options:\n- MacBook Air\n- ThinkPad X1\npin done') == '((send "Here are the options:\\n- MacBook Air\\n- ThinkPad X1") (pin "done"))'
    assert balance_parentheses('send "Plain text version:"\n**Mars** - red planet\nNote: Pluto is a dwarf planet') == '((send "\\\"Plain text version:\\\"\\n**Mars** - red planet\\nNote: Pluto is a dwarf planet"))'
    assert balance_parentheses('(send Here are the planets:\n1. Mercury\n2. Venus)') == '((send "Here are the planets:\\n1. Mercury\\n2. Venus"))'
    assert balance_parentheses('send "hello" world') == '((send "\\"hello\\" world"))'
    assert balance_parentheses('send "Hello"\nHow are you?') == '((send "\\"Hello\\"\\nHow are you?"))'
    # bare "()" lines yield no tokens after _strip_outer_parens and must be skipped, not crash
    assert balance_parentheses('()') == '()'
    assert balance_parentheses('') == '()'
    assert balance_parentheses('   ') == '()'
    assert balance_parentheses('()\nsend hello') == '((send "hello"))'
    assert balance_parentheses('write-file "test.txt" hello\nworld') == '((write-file "test.txt" "hello\\nworld"))'
    assert balance_parentheses('- Found a bug') == '((pin "Found a bug"))'
    assert balance_parentheses('(- Found a bug)') == '((pin "Found a bug"))'
    assert balance_parentheses('- Found\na\nbug') == '((pin "Found\\na\\nbug"))'
    assert balance_parentheses('(- Found a bug') == '((pin "Found a bug"))'

def _format_omega_version(version: str) -> str | None:
    version = version.strip()
    if not version:
        return None
    if version.startswith("Omega version="):
        return version
    if version.startswith("Omega "):
        version = version[len("Omega "):]
    return f"Omega version={version}"


def omega_version(repo_root: str | os.PathLike | None = None) -> str:
    """Return the checkout version, falling back to the baked version file."""
    root = Path(repo_root) if repo_root is not None else Path(projectRootDirectory())

    try:
        # Prevent `git -C` from walking up to a parent repository such as /PeTTa.
        if not (root / ".git").exists():
            raise FileNotFoundError
        result = subprocess.run(
            ["git", "-C", str(root), "describe", "--tags", "--dirty", "--always"],
            check=False,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        )
        if result.returncode == 0:
            version = _format_omega_version(result.stdout)
            if version is not None:
                return version
    except (OSError, subprocess.TimeoutExpired):
        pass

    try:
        version = _format_omega_version(
            (root / "version").read_text(encoding="utf-8")
        )
        if version is not None:
            return version
    except OSError:
        pass

    return "Omega unknown"


def test_omega_version():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        assert omega_version(root) == "Omega unknown"

        (root / "version").write_text("v1.2.3-4-g1234567\n", encoding="utf-8")
        assert omega_version(root) == "Omega version=v1.2.3-4-g1234567"

        (root / "version").write_text("Omega v1.2.3\n", encoding="utf-8")
        assert omega_version(root) == "Omega version=v1.2.3"

if __name__ == "__main__":
    test_omega_version()
