"""Regenerate the four cleaned files for each state.

    python regenerate_release.py                # all 12 states, all 4 views
    python regenerate_release.py --states wi sc # a subset
    python regenerate_release.py --released-only
    python regenerate_release.py --complete-only

Each clean script runs from inside its own state directory, since they all use
paths relative to it. os.remove is shimmed to fall back to truncation for mounts
that allow writes but not unlinks.
"""
import argparse
import os
import runpy
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATES = "ca co ga ky md mt nc ne ok sc wa wi".split()

_real_remove = os.remove
_real_unlink = Path.unlink


def _soft_remove(path, *args, **kwargs):
    try:
        return _real_remove(path, *args, **kwargs)
    except PermissionError:
        open(path, "w").close()


def _soft_unlink(self, *args, **kwargs):
    try:
        return _real_unlink(self, *args, **kwargs)
    except PermissionError:
        self.write_text("")


def run(script: Path) -> bool:
    """Execute a script as __main__ from inside its own directory."""
    cwd = Path.cwd()
    argv = list(sys.argv)
    path = list(sys.path)
    try:
        os.chdir(script.parent)
        sys.argv = [script.name]
        sys.path.insert(0, str(script.parent))
        runpy.run_path(str(script), run_name="__main__")
        return True
    except SystemExit as exc:
        return not exc.code
    except Exception:
        traceback.print_exc()
        return False
    finally:
        os.chdir(cwd)
        sys.argv = argv
        sys.path[:] = path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", nargs="*", default=STATES)
    parser.add_argument("--released-only", action="store_true",
                        help="skip the two complete scripts")
    parser.add_argument("--complete-only", action="store_true",
                        help="skip the two released scripts")
    args = parser.parse_args()

    os.remove = _soft_remove
    Path.unlink = _soft_unlink

    failures = []
    for state in args.states:
        steps = []
        if not args.complete_only:
            steps += [f"{state}_clean_raw.py", f"{state}_clean_full.py"]
        if not args.released_only:
            steps += [f"{state}_clean_complete_raw.py",
                      f"{state}_clean_complete_full.py"]
        for step in steps:
            script = ROOT / state / step
            print(f"\n=== {state}/{step} " + "=" * 40)
            if not script.exists():
                print(f"  missing: {script}")
                failures.append(f"{state}/{step} (missing)")
                break
            if not run(script):
                failures.append(f"{state}/{step}")
                break

    print("\n" + "=" * 60)
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        return 1
    print(f"regenerated {len(args.states)} state(s) cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
