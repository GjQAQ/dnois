from pathlib import Path

from ..base import typing as ty

__all__ = [
    'file_op',
    'readlines',
    'writelines'
]


def file_op(file: ty.TextFile, op: ty.Callable, mode: str = 'r', **kwargs) -> list[str]:
    if isinstance(file, str):
        file = Path(file)
    if isinstance(file, Path):
        with file.open(mode, **kwargs) as f:
            result = op(f)
    else:
        result = op(file)
    return result


def readlines(file: ty.TextFile, mode: str = 'r', encoding='utf-8', **kwargs) -> list[str]:
    return file_op(
        file,
        lambda f: f.readlines(),
        mode,
        encoding=encoding,
        **kwargs
    )


def writelines(file: ty.TextFile, lines: list[str], mode: str = 'w', encoding='utf-8', **kwargs):
    lines = [line + '\n' for line in lines]
    return file_op(
        file,
        lambda f: f.writelines(lines),
        mode,
        encoding=encoding,
        **kwargs
    )
