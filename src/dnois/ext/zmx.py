import dataclasses
import functools
import warnings
from pathlib import Path

import torch

from .. import base, mt, utils
from ..base import typing as ty
from ..optics import rt

__all__ = [
    'agf_readlines',
    'load_agf',
    'sq2zmx',
    'zmx2sq',

    'ZemaxFile',
    'ZemaxParsingError',
]

_zmx_surf_keys = [
    'SSID', 'STOP', 'TYPE', 'FIMP', 'CURV', 'TCED', 'HIDE', 'MIRR', 'SLAB', 'PARM', 'XDAT', 'DISZ', 'GLAS', 'CONI',
    'PZUP', 'DIAM', 'MEMA', 'POPS', 'CLAP',
]
_zmx_surf_key_order = functools.cache(_zmx_surf_keys.index)


def _cmp_zmx_surf_field(f1: str, f2):
    k1, k2 = f1[:4], f2[:4]
    o1, o2 = _zmx_surf_key_order(k1), _zmx_surf_key_order(k2)
    if o1 != o2:
        return o1 - o2
    # same key
    if k1 == 'PARM' or k1 == 'XDAT':
        idx1 = int(f1.split()[1])
        idx2 = int(f2.split()[1])
        return idx1 - idx2
    else:
        return 0


class ZemaxParsingError(RuntimeError):
    """Raised when parsing a ZMX file fails."""
    pass


@dataclasses.dataclass
class ZemaxFile:
    version: str = None
    unit: dict[str, str] = None
    catalogs: list[str] = None
    fov_angles: list[tuple[float, float]] = None
    surfaces: list[list[str]] = dataclasses.field(default_factory=list)
    stop_idx: int = None

    def dump(self, file: ty.TextFile):
        zmx_lines = []
        if self.version is not None:
            zmx_lines.append('VERS ' + self.version)
        zmx_lines.extend([
            'MODE SEQ',
            'NAME Lens from dnois',
        ])
        if self.unit is not None:
            zmx_lines.append('UMIT ' + self.unit['length'])
        if self.catalogs is not None:
            zmx_lines.append('GCAT ' + ' '.join(self.catalogs))
        if self.fov_angles is not None:
            zmx_lines.extend([
                'XFLN ' + ' '.join(map(str, [fov[0] for fov in self.fov_angles])),
                'YFLN ' + ' '.join(map(str, [fov[1] for fov in self.fov_angles])),
            ])
        else:
            zmx_lines.extend(['XFLN 0', 'YFLN 0'])  # FOV must be provided

        for i, s_info in enumerate(self.surfaces):
            zmx_lines.append(f'SURF {i}')

            s_info = s_info.copy()
            if self.stop_idx is not None and self.stop_idx == i:
                s_info.append('STOP')

            s_info = sorted(s_info, key=functools.cmp_to_key(_cmp_zmx_surf_field))
            for field in s_info:
                zmx_lines.append('  ' + field)

        utils.file_op(file, lambda f: f.write(b'\xFF\xFE'), 'wb')
        utils.writelines(file, zmx_lines, 'at', 'utf-16le')

    @classmethod
    def load(cls, file: ty.TextFile) -> ty.Self:
        zmx_lines = utils.readlines(file, encoding='utf-16le')

        obj = cls()
        x_fov = y_fov = None
        current_surf_idx = None
        for line_num, line in enumerate(zmx_lines):
            if len(line.strip()) == 0:
                continue

            if line.startswith('  '):  # with indentation
                if current_surf_idx is None:  # not in the context of a surface
                    continue

                line = line.strip()
                if line.startswith('STOP'):
                    obj.stop_idx = current_surf_idx
                    continue

                obj.surfaces[current_surf_idx].append(line)
            elif line.startswith('VERS'):
                obj.version = line[5:].strip()
            elif line.startswith('MODE'):
                mode = line[5:].strip()
                if mode != 'SEQ':
                    raise ZemaxParsingError(f'Cannot parse .zmx file with mode={mode}')
            elif line.startswith('UNIT'):
                units = line.split()[1:]
                obj.unit = {'length': units[0]}
            elif line.startswith('GCAT'):
                obj.catalogs = line.split()[1:]
            elif line.startswith('XFLN'):
                x_fov = map(float, line.split()[1:])
            elif line.startswith('YFLN'):
                y_fov = map(float, line.split()[1:])
            elif line.startswith('SURF'):  # start of a surface
                current_surf_idx = int(line[5:])
                while len(obj.surfaces) <= current_surf_idx:  # use loop to handle out-of-order
                    obj.surfaces.append([])
            # else:
            #     raise ZemaxParsingError(f'Unexpected field {line[:4]} in line {line_num}')

        if x_fov is not None:
            obj.fov_angles = list(zip(x_fov, y_fov))

        return obj


class ZemaxSurfaceConverter:
    name: str
    type: type[rt.Surface]
    default_kwargs = {}

    def __init__(self, unit: str):
        self.unit = unit

    def dump(self, surface: rt.Surface) -> list[str]:
        fields = [
            f'TYPE {self.zmx_name()}',
            f'FIMP',
            f'DISZ {self.icl(surface.distance):.15E}',
        ]

        apt = surface.aperture
        fields.append(f'DIAM {self.icl(apt.max_radius()):.15E} 0 0 0 1 ""')
        if isinstance(apt, rt.AnnularAperture):
            fields.append(f'CLAP {self.icl(apt.r1):.15E} {self.icl(apt.r2):.15E} 0')

        if surface.reflective:
            fields.append(f'GLAS MIRROR')
        elif surface.material.name != 'air':
            fields.append(f'GLAS {surface.material.primitive_name}')
        return fields

    def parse(self, fields: list[str]) -> rt.Surface:
        kwargs = self.default_kwargs.copy()
        for field_line in fields:
            field_line = field_line.split()
            field_name, field_value = field_line[0], field_line[1:]
            self.handle_field(field_name, field_value, kwargs)
        return self.type(**kwargs)

    def handle_field(self, key: str, values: list[str], kwargs: dict):
        # modify kwargs dict in place
        if key == 'CLAP':
            kwargs['aperture'] = rt.AnnularAperture(self.cl(values[0]), self.cl(values[1]))
        elif key == 'DIAM':
            kwargs.setdefault('aperture', self.cl(values[0]))
        elif key == 'DISZ':
            kwargs['d'] = self.cl(values[0])
        elif key == 'GLAS':
            if values[0] == 'MIRROR':
                kwargs['reflective'] = True
            else:
                kwargs['material'] = values[0]

    def cl(self, value: float | str):  # convert length
        if isinstance(value, str):
            value = float(value)
        return base.Length.as_default(value, self.unit)

    def icl(self, value: float | ty.Ts) -> float:  # inverse convert length
        if torch.is_tensor(value):
            value = value.item()
        return base.Length.default_to(value, self.unit)

    @classmethod
    def zmx_name(cls) -> str:
        return cls.name

    @classmethod
    @ty.overload
    def create(cls, name: str, *args, **kwargs):
        subclasses = utils.subclasses(cls)
        for subclass in subclasses:
            if subclass.name == name:
                return subclass(*args, **kwargs)
        raise ZemaxParsingError(f'Unknown Zemax surface type: {name}')

    @classmethod
    @ty.overload
    def create(cls, stype, *args, **kwargs):
        subclasses = utils.subclasses(cls)
        for subclass in subclasses:
            if subclass.type == stype:
                return subclass(*args, **kwargs)
        raise ZemaxParsingError(f'Unknown surface type: {stype.__name__}')

    @classmethod
    @ty.final
    def create(cls, name_or_type, *args, **kwargs):
        ols = ty.get_overloads(cls.create)
        if isinstance(name_or_type, str):
            return ols[0](cls, name_or_type, *args, **kwargs)
        else:
            return ols[1](cls, name_or_type, *args, **kwargs)


class DumpOnlyConverter(ZemaxSurfaceConverter):
    name = ''
    type = ...

    def handle_field(self, *args, **kwargs):
        raise RuntimeError(f'Unexpected calling')


class DGratingConverter(ZemaxSurfaceConverter):
    name = 'DGRATING'
    type = rt.Grating

    def dump(self, surface: rt.Grating) -> list[str]:
        fields = super().dump(surface)

        period_inv = 1.0 / base.Length.default_to(surface.period.item(), 'um')
        fields.append(f'PARM 1 {period_inv:.15E}')

        order = surface.orders[0]
        fields.append(f'PARM 2 {order}')

        return fields

    def handle_field(self, key: str, values: list[str], kwargs: dict):
        super().handle_field(key, values, kwargs)
        if key == 'PARM':
            param_n, param_v = values[0], values[1]
            if param_n == '1':
                kwargs['period'] = base.Length.as_default(1 / float(param_v), 'um')
            elif param_n == '2':
                order = int(param_v)
                kwargs['orders'] = (order, order)


class ParaxialConverter(ZemaxSurfaceConverter):
    name = 'PARAXIAL'
    type = rt.ThinLens

    def dump(self, surface: rt.ThinLens) -> list[str]:
        fields = super().dump(surface)

        fields.append(f'PARM 1 {self.icl(surface.fl1):.15E}')

        return fields

    def handle_field(self, key: str, values: list[str], kwargs: dict):
        super().handle_field(key, values, kwargs)
        kwargs['fl_equal'] = True
        if key == 'PARM':
            param_n, param_v = values[:2]
            if param_n == '1':
                kwargs['fl1'] = self.cl(param_v)


class StandardConverter(ZemaxSurfaceConverter):
    name = 'STANDARD'
    type = rt.Conic

    def dump(self, surface: rt.Conic) -> list[str]:
        fields = super().dump(surface)

        # Add curvature (radius of curvature)
        if surface.roc == float('inf'):
            curv = 0.0  # Infinite radius of curvature
        else:
            curv = 1.0 / self.icl(surface.roc)
        fields.append(f'CURV {curv:.15E} 0 0 0 0 ""')

        # Add conic constant
        fields.append(f'CONI {surface.conic.item():.15E}')

        return fields

    def handle_field(self, key: str, values: list[str], kwargs: dict):
        super().handle_field(key, values, kwargs)
        if key == 'CURV':
            try:
                roc = 1. / float(values[0])
            except ZeroDivisionError:
                roc = float('inf')
            kwargs['roc'] = self.cl(roc)
        elif key == 'CONI':
            kwargs['conic'] = float(values[0])


class StandardAsCircularStopConverter(DumpOnlyConverter):
    name = ''
    type = rt.CircularStop

    def dump(self, surface: rt.CircularStop) -> list[str]:
        fields = super().dump(surface)
        fields.append('CURV 0 0 0 0 0 ""')
        fields.append('CONI 0')
        return fields

    @classmethod
    def zmx_name(cls) -> str:
        return 'STANDARD'


class EvenAsphConverter(StandardConverter):
    name = 'EVENASPH'
    type = rt.EvenAspherical
    aspheric_attr_name = 'coefficients'

    def dump(self, surface: rt.EvenAspherical) -> list[str]:
        fields = super().dump(ty.cast(rt.Conic, surface))

        # Add aspheric coefficients
        coefficients = getattr(surface, self.aspheric_attr_name)
        ratio = base.Length.default() / base.Length.from_str(self.unit)

        for idx, coef in enumerate(coefficients):
            # note that PARM 1 is the first coefficient
            zmx_value = coef.item() / (ratio ** (2 * idx + 1))
            fields.append(f'PARM {idx + 1} {zmx_value:.15E}')

        return fields

    def handle_field(self, key: str, values: list[str], kwargs: dict):
        super().handle_field(key, values, kwargs)
        if key == 'PARM':
            idx, value = int(values[0]), float(values[1])
            if idx == 0:
                return  # unused parameter
            if value == 0.:
                return

            kwargs.setdefault(self.aspheric_attr_name, [])
            c = kwargs[self.aspheric_attr_name]
            while len(c) < idx:  # note that PARM 1 is the first coefficient
                c.append(0.)
            ratio = base.Length.default() / base.Length.from_str(self.unit)
            c[idx - 1] = value * ratio ** (2 * idx - 1)


class Binary2Converter(EvenAsphConverter):
    name = 'BINARY_2'
    type = rt.AsphericalRadialPhase

    def dump(self, surface: rt.AsphericalRadialPhase) -> list[str]:
        fields = super().dump(surface)

        # Add diffraction order (always 1 for supported surfaces)
        fields.append('PARM 0 1')

        # Add phase coefficients
        # Add number of terms
        fields.append(f'XDAT 1 {surface.phase_items} 0 0 1 0 0 ""')

        # Add normalization radius
        fields.append(f'XDAT 2 {self.icl(surface.norm_radius):.15E} 0 0 1 0 0 ""')

        # Add phase coefficients
        for idx, coef in enumerate(surface.phase_coefficients):
            fields.append(f'XDAT {idx + 3} {coef.item():.15E} 0 0 1 0 0 ""')

        return fields

    def handle_field(self, key: str, values: list[str], kwargs: dict):
        super().handle_field(key, values, kwargs)
        if key == 'PARM':
            idx, value = values[:2]
            if idx == '0':  # diffraction order
                if value != '1':
                    raise ZemaxParsingError(f'Binary2 surface with diffraction order {value} is not supported')
        elif key == 'XDAT':
            idx, value = values[:2]
            if idx == '1':
                return  # number of terms
            elif idx == '2':
                kwargs['norm_radius'] = self.cl(value)
            else:
                # phase coefficients
                kwargs.setdefault('phase_coef', [])
                c = kwargs['phase_coef']
                idx = int(idx) - 3  # XDAT 3 is the first coefficient
                while len(c) <= idx:
                    c.append(0.)
                c[idx] = float(value)


class SzernsagConverter(EvenAsphConverter):
    name = 'SZERNSAG'
    type = rt.Zernike
    aspheric_attr_name = 'a'

    def dump(self, surface: rt.Zernike) -> list[str]:
        fields = super().dump(ty.cast(rt.EvenAspherical, surface))

        # Add Zernike coefficients
        # Add number of terms
        fields.append(f'XDAT 1 {surface.z_n} 0 0 1 0 0 ""')

        # Add normalization radius
        fields.append(f'XDAT 2 {self.icl(surface.norm_radius):.15E} 0 0 1 0 0 ""')

        # Add Zernike coefficients
        for idx, coef in enumerate(surface.z):
            fields.append(f'XDAT {idx + 3} {coef.item():.15E} 1 0 1 0 0 ""')

        return fields

    def handle_field(self, key: str, values: list[str], kwargs: dict):
        super().handle_field(key, values, kwargs)
        if key == 'XDAT':
            idx, value = values[:2]
            if idx == '1':
                return  # number of terms
            elif idx == '2':
                kwargs['norm_radius'] = self.cl(value)
            else:
                # phase coefficients
                kwargs.setdefault('z', [])
                c = kwargs['z']
                idx = int(idx) - 3  # XDAT 3 is the first coefficient
                while len(c) <= idx:
                    c.append(0.)
                c[idx] = float(value)


class FresnelsConverter(EvenAsphConverter):
    name = 'FRESNELS'
    type = rt.Fresnel
    default_kwargs = {'wrapping': 0}


def zmx2sq(file: str | Path | ty.TextIO) -> rt.CoaxialSurfaceSequence:
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
    zmx = ZemaxFile.load(file)

    zmx_sq = zmx.surfaces[1:-1]  # discard object and image plane
    sq = [_surface_from_zmx_fields(surf_fields, i, zmx.unit['length'].lower()) for i, surf_fields in enumerate(zmx_sq)]
    sq = rt.CoaxialSurfaceSequence(sq, stop_idx=zmx.stop_idx)
    return sq


def _surface_from_zmx_fields(fields: list[str], idx: int, unit: str) -> rt.Surface:
    for field_line in fields:
        if field_line[:4] == 'TYPE':
            stype = field_line.split()[1]
            break
    else:
        raise ZemaxParsingError(f'Type not found for surface {idx}')

    parser = ZemaxSurfaceConverter.create(stype, unit)
    surf = parser.parse(fields)
    return surf


def sq2zmx(
    file: ty.TextFile,
    sq: rt.CoaxialSurfaceSequence,
    version: str = None,
    unit: dict[str, str] = None,
):
    if unit is None:
        unit = {'length': 'MM'}

    zmx = ZemaxFile(
        version=version,
        unit=unit,
        surfaces=[],
        stop_idx=sq.stop_idx,
    )

    zmx.surfaces.append([
        'TYPE STANDARD',
        'FIMP',
        'CURV 0.0 0 0 0 0 ""',
        'DISZ INFINITY',
        'DIAM 0 0 0 0 1 ""',
    ])  # object plane
    for surface in sq:
        parser = ZemaxSurfaceConverter.create(surface.__class__, unit['length'].lower())
        zmx.surfaces.append(parser.dump(surface))
    zmx.surfaces.append([
        'TYPE STANDARD',
        'FIMP',
        'CURV 0.0 0 0 0 0 ""',
        'DISZ 0',
        'DIAM 0 0 0 0 1 ""',
    ])  # image plane

    zmx.dump(file)


def load_agf(
    file: str | Path | ty.TextIO,
    qualifier: str = None,
    modifier: ty.Callable[[mt.Material], mt.Material] = None,
    existed_behavior: ty.Literal['skip', 'overwrite', 'warn', 'error'] = 'warn',
):
    if isinstance(file, str):
        file = Path(file)
    if isinstance(file, Path):
        agf_lines = agf_readlines(file)
        if qualifier is None:
            qualifier = file.stem.upper()
    else:
        agf_lines = file.readlines()
        if qualifier is None:
            qualifier = ''

    material_list = list(_split_agf(agf_lines))
    material_list = map(functools.partial(_construct_material, qualifier=qualifier), material_list)
    material_list = map(modifier, material_list) if modifier is not None else material_list
    for material in material_list:
        if mt.registered(material.name):
            if existed_behavior != 'overwrite':
                if existed_behavior == 'warn':
                    warnings.warn(f'Material {material.name} already exists, skip')
                if existed_behavior != 'error':
                    continue
        mt.register(material, existed_behavior != 'error')


def agf_readlines(path: str | Path):
    path = Path(path)

    with path.open('rb') as f:
        bom = f.read(4)
    if bom.startswith(b'\xFF\xFE'):
        encoding='utf-16le'
    else:
        encoding='utf-8'

    with path.open('r', encoding=encoding, errors='replace') as f:
        return f.readlines()


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
    mt.Conrady,
    mt.Sellmeier3,
    None,
    None,
    mt.Sellmeier4,
    None,
    mt.Sellmeier5,
    None,
    None,
]


def _construct_material(fields: list[str], qualifier: str) -> mt.Material:
    nm_line = _split_field(fields, 'NM')
    name, dispersion_id = nm_line[1], int(round(float(nm_line[2])))
    if dispersion_id >= len(_dispersion_list) or _dispersion_list[dispersion_id] is None:
        raise ZemaxParsingError(f'Undefined dispersion identifier: {dispersion_id}')
    name = f'{qualifier}:{name}'
    cls = _dispersion_list[dispersion_id]

    cd_line = _split_field(fields, 'CD')
    c = [_float(param) for param in cd_line[1:]]

    td_line = _split_field(fields, 'TD')
    td = [_float(param) for param in td_line[1:]]
    ref_t = _get_thermal_data(6, td, 20.)
    d0 = _get_thermal_data(0, td)
    d1 = _get_thermal_data(1, td)
    d2 = _get_thermal_data(2, td)
    e0 = _get_thermal_data(3, td)
    e1 = _get_thermal_data(4, td)
    ltk = _get_thermal_data(5, td)

    ld_line = _split_field(fields, 'LD')
    min_wl = float(ld_line[1])
    max_wl = float(ld_line[2])

    common_args = (min_wl, max_wl, ref_t, d0, d1, d2, e0, e1, ltk)

    if cls == mt.Schott:
        obj = cls(name, c[:6], *common_args)
    elif cls == mt.Sellmeier1:
        obj = cls(name, c[:6:2], c[1:6:2], *common_args)  # noqa
    elif cls == mt.Herzberger:
        obj = cls(name, c[:6], *common_args)
    elif cls == mt.Conrady:
        obj = cls(name, *c[:3], *common_args)
    elif cls == mt.Sellmeier3:
        obj = cls(name, c[:8:2], c[1:8:2], *common_args)  # noqa
    elif cls == mt.Sellmeier4:
        obj = cls(name, *c[:5], *common_args)
    elif cls == mt.Sellmeier5:
        obj = cls(name, c[:10:2], c[1:10:2], *common_args)  # noqa
    else:
        raise RuntimeError('Unexpected error')
    return obj


def _get_thermal_data(i, td, default=None):
    return td[i] if len(td) > i else default


def _split_field(fields, tag):
    try:
        return next(filter(lambda line: line.startswith(tag), fields)).split()
    except StopIteration:
        return [tag]


def _float(s, default=0.):
    try:
        return float(s)
    except ValueError:
        return default
