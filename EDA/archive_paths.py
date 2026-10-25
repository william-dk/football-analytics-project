"""Resolve historical audit paths after the explicit EDA legacy migration.

Existing active files always win. Only missing paths inside EDA may map to
the matching relative path under EDA/legacy; unavailable inputs fail loudly.
Historical JSON/CSV reports keep their original bytes and provenance paths.
"""
from pathlib import Path

EDA_ROOT = Path(__file__).resolve().parent


def resolve_recorded_path(path, eda_root=EDA_ROOT):
    eda_root = Path(eda_root).resolve()
    original = Path(path)
    if not original.is_absolute():
        original = eda_root.parent / original
    original = original.resolve()
    if original.exists():
        return original
    try:
        relative = original.relative_to(eda_root)
    except ValueError:
        raise FileNotFoundError(f'Historical input is unavailable: {original}') from None
    if not relative.parts or relative.parts[0] == 'legacy':
        raise FileNotFoundError(f'Historical input is unavailable: {original}')
    archived = eda_root / 'legacy' / relative
    if archived.exists():
        return archived
    raise FileNotFoundError(f'Input is missing from both active and legacy locations: {original}')
