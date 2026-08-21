"""Load optional API tokens from a gitignored keys.json into the environment.

An already-set variable is never overwritten, a missing or malformed file is not
an error, and values are never printed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

KEYS_PATH = Path(__file__).resolve().parent / "keys.json"

# Allow-list: nothing outside this tuple is ever exported.
ALLOWED = ("TABPFN_TOKEN", "HF_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")


def load_keys(path: "Path | str | None" = None, *, verbose: bool = False) -> list[str]:
    """Export allow-listed keys that are not already set; return their names."""
    p = Path(path) if path is not None else KEYS_PATH
    try:
        data = json.loads(p.read_text())
    except FileNotFoundError:
        return []
    except (json.JSONDecodeError, OSError) as e:
        print(f"[keys] ignoring {p.name}: {type(e).__name__}: {e}")
        return []

    exported = []
    for name in ALLOWED:
        val = data.get(name)
        if val and not os.environ.get(name):
            os.environ[name] = str(val)
            exported.append(name)
    if verbose and exported:
        print(f"[keys] loaded from {p.name}: {', '.join(exported)}")
    return exported


def loaded_key_names() -> list[str]:
    """Names of the allow-listed variables currently set."""
    return [n for n in ALLOWED if os.environ.get(n)]


if __name__ == "__main__":
    got = load_keys(verbose=True)
    print(f"exported: {got or 'nothing (already set, or no keys.json)'}")
    print(f"present now: {loaded_key_names() or 'none'}")
