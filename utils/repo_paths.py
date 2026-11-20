"""Location-independent fingerprints for state saved inside this repository.

Absolute paths inside the checkout are fingerprinted relative to it, so the
whole repository can be moved without invalidating saved memory state. States
saved before this rule used absolute paths; they remain reusable at their
original location, or after a move when the former root is declared in
MEMORYDATA_LEGACY_REPO_ROOTS (os.pathsep-separated).
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_PREFIX = "<memorydata>/"
LEGACY_ROOTS_ENV = "MEMORYDATA_LEGACY_REPO_ROOTS"


def repo_relative(value, root=None):
    """POSIX path of an absolute path inside the repository, otherwise None."""
    if not isinstance(value, str) or not value or len(value) > 4096 or "\n" in value:
        return None
    try:
        path = Path(value)
        if not path.is_absolute():
            return None
        return path.resolve().relative_to(root or REPO_ROOT).as_posix()
    except (OSError, ValueError):
        return None


def map_repo_paths(value, convert):
    """Apply convert(original, relative_path) to every absolute in-repository path string."""
    if isinstance(value, dict):
        return {key: map_repo_paths(item, convert) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [map_repo_paths(item, convert) for item in value]
    relative = repo_relative(value)
    return value if relative is None else convert(value, relative)


def portable(value):
    """Replace in-repository absolute paths with a root-independent marker."""
    return map_repo_paths(value, lambda original, relative: REPO_PREFIX + relative)


def legacy_repo_roots():
    return [root.strip() for root in os.environ.get(LEGACY_ROOTS_ENV, "").split(os.pathsep) if root.strip()]


def _under_root(original, relative, root):
    """Swap only the leading checkout root, keeping the rest of the string byte for byte.

    Runtime paths can mix separators (os.path.join onto a configured root), so
    rebuilding them from the relative path alone would not reproduce old fingerprints.
    """
    for current, former in ((str(REPO_ROOT), str(Path(root))), (REPO_ROOT.as_posix(), Path(root).as_posix())):
        head, tail = original[:len(current)], original[len(current):]
        if head.lower() == current.lower() and tail[:1] in ("", "\\", "/"):
            return former + tail
    return str(Path(root).joinpath(*relative.split("/")))


def legacy_variants(value):
    """value as old absolute-path fingerprints saw it: unchanged, or under each declared former root."""
    return [value] + [map_repo_paths(value, lambda original, relative, root=root: _under_root(original, relative, root))
                      for root in legacy_repo_roots()]
