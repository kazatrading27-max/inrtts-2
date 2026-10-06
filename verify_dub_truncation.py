"""Proof that the X-Dub-Truncated change is output-preserving.

Extracts ``_assemble_dub`` from the current ``api_server.py`` and from the
pre-change revision, runs both over identical inputs, and asserts:

  * the returned AUDIO is byte-identical (``np.array_equal``) in every scenario;
  * the new truncation COUNT matches what the pre-change code logged as a warning.

The "before" revision is taken from the newest ``api_server.py.bak-*`` snapshot
if one exists, otherwise from ``git show HEAD:api_server.py``.

Run it directly (it is not a pytest module, so pytest will not collect it):

    python verify_dub_truncation.py

Needs numpy only - no model, no GPU, no server.
"""
import ast
import glob
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CURRENT = os.path.join(ROOT, "api_server.py")
SR = 8000


class _Log:
    def __init__(self):
        self.warnings = []

    def warning(self, msg):
        self.warnings.append(msg)

    def info(self, *a):
        pass

    def error(self, *a):
        pass


def _before_source() -> str:
    backups = sorted(glob.glob(os.path.join(ROOT, "api_server.py.bak-*")))
    if backups:
        return open(backups[-1], encoding="utf-8").read()
    try:
        out = subprocess.run(
            ["git", "show", "HEAD:api_server.py"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"Could not obtain a pre-change revision: {exc}")
        sys.exit(2)


def _extract(src: str, name: str) -> str:
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise SystemExit(f"{name} not found")


def _load(src: str):
    log = _Log()
    ns = {"np": np, "List": list, "Tuple": tuple, "logger": log}
    exec(compile(src, "<extracted>", "exec"), ns)
    return ns["_assemble_dub"], log


def _clip(start_ms, end_ms, dur_ms):
    n = int(SR * dur_ms / 1000)
    return ({"start_ms": start_ms, "end_ms": end_ms},
            (np.arange(n, dtype=np.int32) % 30000).astype(np.int16))


SCENARIOS = {
    "no overrun":     ([_clip(0, 2000, 1500), _clip(2000, 4000, 1500)], 200, False, 0),
    "one overrun":    ([_clip(0, 1000, 1500), _clip(1000, 3000, 1000)], 200, False, 1),
    "all overrun":    ([_clip(0, 500, 2000), _clip(500, 1000, 2000)], 200, False, 2),
    "reset timeline": ([_clip(0, 1000, 3000), _clip(1000, 2000, 3000)], 200, True, 0),
    "missing stamps": ([({"start_ms": None, "end_ms": None}, np.ones(800, dtype=np.int16)),
                        ({"start_ms": None, "end_ms": None}, np.ones(800, dtype=np.int16))], 200, False, 0),
    "zero-width slot": ([_clip(0, 1000, 1000), _clip(1000, 1000, 1000)], 200, False, 1),
    "single clip":    ([_clip(0, 1000, 500)], 0, False, 0),
}


def main() -> int:
    old_fn, old_log = _load(_extract(_before_source(), "_assemble_dub"))
    new_fn, new_log = _load(_extract(open(CURRENT, encoding="utf-8").read(), "_assemble_dub"))

    failures = 0
    for name, (clips, gap, reset, expected) in SCENARIOS.items():
        c_old = [(dict(s), w.copy()) for s, w in clips]
        c_new = [(dict(s), w.copy()) for s, w in clips]
        old_log.warnings.clear()
        new_log.warnings.clear()

        audio_old = old_fn(c_old, SR, gap, reset)
        audio_new, count = new_fn(c_new, SR, gap, reset)

        logged_old = int(old_log.warnings[0].split()[1]) if old_log.warnings else 0
        logged_new = int(new_log.warnings[0].split()[1]) if new_log.warnings else 0

        ok = (np.array_equal(audio_old, audio_new)
              and audio_old.shape == audio_new.shape
              and count == expected
              and logged_old == expected
              and logged_new == expected)
        failures += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {name:<16} audio_identical={np.array_equal(audio_old, audio_new)} "
              f"count={count} (expected {expected}) old_log={logged_old} new_log={logged_new}")

    print(f"\nscenarios: {len(SCENARIOS)}  failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
