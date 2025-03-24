from pathlib import Path

from .. import base
from ..base import typing
from ..optics import rt

__all__ = [
    'slist_from_zmx',

    'ZmxParsingError',
]


class ZmxParsingError(RuntimeError):
    """Raised when parsing a ZMX file fails."""
    pass


def slist_from_zmx(file: str | Path | typing.TextIO) -> rt.CoaxialSurfaceList:
    if isinstance(file, str):
        file = Path(file)
    if isinstance(file, Path):
        with file.open('r', encoding='utf-16le') as f:
            zmx_lines = f.readlines()
    else:
        zmx_lines = file.readlines()

    zmx_surf_list: list[list[list[str]]] = []  # innermost list[str] is actually tuple[str, str]
    current_surf_idx: int | None = None
    stop_idx: int | None = None
    zmx_lens_unit: str = 'mm'
    for line_num, line in enumerate(zmx_lines):
        if len(line.strip()) == 0:
            continue

        if line.startswith('  '):  # with indentation
            if current_surf_idx is None:  # not in the context of a surface
                continue

            line = line.strip()
            if line.startswith('STOP'):
                stop_idx = current_surf_idx
                continue

            if len(line) > 4 and line[4] != ' ':
                raise ZmxParsingError(f'Undefined format({line_num}): {line}')
            zmx_surf_list[current_surf_idx].append(line.split(' ', 1))
        else:  # without indentation
            if line.startswith('SURF'):  # start of a surface
                current_surf_idx = int(line[5:])
                while len(zmx_surf_list) <= current_surf_idx:  # use loop to handle out-of-order
                    zmx_surf_list.append([])
            else:  # not a surface
                current_surf_idx = None
                if line.startswith('UNIT'):
                    zmx_lens_unit = line.split(' ', 2)[1].lower()

    zmx_surf_list = zmx_surf_list[1:-1]  # discard object and image plane
    slist = [_surface_from_zmx_segment(segment, i, zmx_lens_unit) for i, segment in enumerate(zmx_surf_list)]
    slist = rt.CoaxialSurfaceList(slist, stop_idx=stop_idx)
    return slist


def _surface_from_zmx_segment(segments: list[list[str]], idx: int, unit: str) -> rt.Surface:
    for segment in segments:
        if segment[0] == 'TYPE':
            stype = segment[1]
            break
    else:
        raise ZmxParsingError(f'Type not found for surface {idx}')

    if stype == 'PARAXIAL':
        surf = rt.ThinLens(**_parse_surface_args(segments, _parse_thin_lens_segment, unit), fl_equal=True)
    elif stype == 'STANDARD':
        surf = rt.Conic(**_parse_surface_args(segments, _parse_conic_segment, unit))
    else:
        raise ZmxParsingError(f'Undefined surface type: {stype}')
    return surf


def _parse_surface_args(segments: list[list[str]], segment_parser, unit: str) -> dict[str, typing.Any]:
    args = {}
    for segment in segments:
        if len(segment) != 2:
            continue
        key, value = segment_parser(segment, unit)
        if key is not None:
            args[key] = value
    return args


def _parse_thin_lens_segment(segment: list[str], unit: str) -> tuple[str | None, typing.Any]:
    key, value = segment
    if key == 'PARM':
        param_n, param_v = value.split(' ', 1)
        if param_n == '1':
            return 'fl1', base.Length.as_default(float(param_v), unit)
        else:
            return None, None
    else:
        return _parse_segment_common(segment, unit)


def _parse_conic_segment(segment: list[str], unit: str) -> tuple[str | None, typing.Any]:
    key, value = segment
    if key == 'CURV':
        try:
            roc = 1. / float(value.split(' ', 1)[0])
        except ZeroDivisionError:
            roc = float('inf')
        return 'roc', base.Length.as_default(roc, unit)
    else:
        return _parse_segment_common(segment, unit)


def _parse_segment_common(segment: list[str], unit: str) -> tuple[str | None, typing.Any]:
    key, value = segment
    if key == 'GLAS':
        return 'material', value.split(' ', 1)[0]
    elif key == 'DIAM':
        r = float(value.split(' ', 1)[0])
        aperture = rt.CircularAperture(base.Length.as_default(r, unit) * 2)
        return 'aperture', aperture
    elif key == 'DISZ':
        return 'd', base.Length.as_default(float(value), unit)
    else:
        return None, None
