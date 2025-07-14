import warnings
from pathlib import Path

from .. import base, mt
from ..base import typing as ty
from ..optics import rt

__all__ = [
    'load_agf',
    'slist_from_zmx',

    'ZemaxParsingError',
]


class ZemaxParsingError(RuntimeError):
    """Raised when parsing a ZMX file fails."""
    pass


def slist_from_zmx(file: str | Path | ty.TextIO) -> rt.CoaxialSurfaceSequence:
    """
    Parse a ZMX file and return a :class:`~dnois.optics.rt.CoaxialSurfaceSequence` object.

    .. warning::
        This function is experimental and is subject to change.

    :param file: The ZMX file to be parsed. Can be either a file path (``str`` or ``pathlib.Path``),
        or a file-like object (implementing ``readlines()``).
    :type file: str | Path | ty.TextIO
    :return: A :class:`~dnois.optics.rt.CoaxialSurfaceSequence` object.
    :rtype: ~dnois.optics.rt.CoaxialSurfaceSequence
    """
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
                raise ZemaxParsingError(f'Undefined format({line_num}): {line}')
            # one line in a surface context, example: TYPE STANDARD
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
    slist = rt.CoaxialSurfaceSequence(slist, stop_idx=stop_idx)
    return slist


def _surface_from_zmx_segment(segments: list[list[str]], idx: int, unit: str) -> rt.Surface:
    for segment in segments:
        if segment[0] == 'TYPE':
            stype = segment[1]
            break
    else:
        raise ZemaxParsingError(f'Type not found for surface {idx}')

    if stype == 'DGRATING':
        surf = rt.Grating(**_parse_surface_args(segments, _parse_grating_segment, unit))
    elif stype == 'PARAXIAL':
        surf = rt.ThinLens(**_parse_surface_args(segments, _parse_thin_lens_segment, unit), fl_equal=True)
    elif stype == 'STANDARD':
        surf = rt.Conic(**_parse_surface_args(segments, _parse_conic_segment, unit))
    else:
        raise ZemaxParsingError(f'Undefined surface type: {stype}')
    return surf


def _parse_surface_args(segments: list[list[str]], segment_parser, unit: str) -> dict[str, ty.Any]:
    args = {}
    for segment in segments:
        if len(segment) != 2:
            continue
        key, value = segment_parser(segment, unit)
        if key is not None:
            args[key] = value
    return args


def _parse_conic_segment(segment: list[str], unit: str) -> tuple[str | None, ty.Any]:
    key, value = segment
    if key == 'CURV':
        try:
            roc = 1. / float(value.split(' ', 1)[0])
        except ZeroDivisionError:
            roc = float('inf')
        return 'roc', base.Length.as_default(roc, unit)
    else:
        return _parse_segment_common(segment, unit)


def _parse_grating_segment(segment: list[str], unit: str) -> tuple[str | None, ty.Any]:
    key, value = segment
    if key == 'PARM':
        param_n, param_v = value.split(' ', 1)
        if param_n == '1':
            return 'period', base.Length.as_default(1 / float(param_v), 'um')
        elif param_n == '2':
            order = int(param_v)
            return 'orders', (order, order)
        else:
            return None, None
    else:
        return _parse_segment_common(segment, unit)


def _parse_thin_lens_segment(segment: list[str], unit: str) -> tuple[str | None, ty.Any]:
    key, value = segment
    if key == 'PARM':
        param_n, param_v = value.split(' ', 1)
        if param_n == '1':
            return 'fl1', base.Length.as_default(float(param_v), unit)
        else:
            return None, None
    else:
        return _parse_segment_common(segment, unit)


def _parse_segment_common(segment: list[str], unit: str) -> tuple[str | None, ty.Any]:
    key, value = segment
    if key == 'GLAS':
        return 'material', value.split(' ', 1)[0]
    elif key == 'DIAM':
        r = float(value.split(' ', 1)[0])
        aperture = rt.CircularAperture(base.Length.as_default(r, unit))
        return 'aperture', aperture
    elif key == 'DISZ':
        return 'd', base.Length.as_default(float(value), unit)
    else:
        return None, None


def load_agf(
    file: str | Path | ty.TextIO,
    modifier: ty.Callable[[mt.Material], mt.Material] = None,
    existed_behavior: ty.Literal['skip', 'overwrite', 'warn', 'error'] = 'warn',
):
    if isinstance(file, str):
        file = Path(file)
    if isinstance(file, Path):
        with file.open('r', encoding='utf-8') as f:
            agf_lines = f.readlines()
    else:
        agf_lines = file.readlines()

    material_list = list(_split_agf(agf_lines))
    material_list = map(_construct_material, material_list)
    material_list = map(modifier, material_list) if modifier is not None else material_list
    for material in material_list:
        if mt.registered(material.name):
            if existed_behavior != 'overwrite':
                if existed_behavior == 'warn':
                    warnings.warn(f'Material {material.name} already exists, skip')
                if existed_behavior != 'error':
                    continue
        mt.register(material, existed_behavior != 'error')


def _split_agf(agf_lines: list[str]):
    group = []
    for line in agf_lines:
        if line.startswith('NM'):
            if group:
                yield group
            group = [line]
        elif group:
            group.append(line)
    if group:
        yield group


_dispersion_list = [
    None,
    mt.Schott,
    mt.Sellmeier1,
    mt.Herzberger,
    None,
    None,
    mt.Sellmeier3,
    None,
    None,
    mt.Sellmeier4,
    None,
    mt.Sellmeier5,
    None,
    None,
]


def _construct_material(fields: list[str]) -> mt.Material:
    nm_line = _split_field(fields, 'NM')
    name, dispersion_id = nm_line[1], int(round(float(nm_line[2])))
    if dispersion_id >= len(_dispersion_list) or _dispersion_list[dispersion_id] is None:
        raise ZemaxParsingError(f'Undefined dispersion identifier: {dispersion_id}')
    cls = _dispersion_list[dispersion_id]

    cd_line = _split_field(fields, 'CD')
    c = [_float(param) for param in cd_line[1:]]

    ld_line = _split_field(fields, 'LD')
    min_wl = float(ld_line[1])
    max_wl = float(ld_line[2])

    if cls == mt.Schott:
        obj = cls(name, c[:6], min_wl, max_wl)
    elif cls == mt.Sellmeier1:
        obj = cls(name, c[:6:2], c[1:6:2], min_wl, max_wl)  # noqa
    elif cls == mt.Herzberger:
        obj = cls(name, c[:6], min_wl, max_wl)
    elif cls == mt.Sellmeier3:
        obj = cls(name, c[:8:2], c[1:8:2], min_wl, max_wl)  # noqa
    elif cls == mt.Sellmeier4:
        obj = cls(name, *c[:5], min_wl, max_wl)
    elif cls == mt.Sellmeier5:
        obj = cls(name, c[:10:2], c[1:10:2], min_wl, max_wl)  # noqa
    else:
        raise RuntimeError('Unexpected error')
    return obj


def _split_field(fields, tag):
    return next(filter(lambda line: line.startswith(tag), fields)).split()


def _float(s, default=0.):
    try:
        return float(s)
    except ValueError:
        return default
