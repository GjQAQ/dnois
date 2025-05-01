from enum import Enum
from functools import cache
import math

from . import typing

__all__ = [
    'convert',
    'get_default',
    'set_default',

    'Unit',
    'Length',
    'Angle',
]

float_print_fmt: str = '.6g'


def fmt(v: float) -> str:  # this function is not public and hence without docstring
    s = f'{{:{float_print_fmt}}}'
    return s.format(v)


class Unit(Enum):
    """
    Base class of all units.

    :param str symbol: Symbol of this unit, e.g. ``'m'`` for meter.
    :param float scale: Scale factor to convert this unit to another unit of the same type.
    """

    def __init__(self, symbol: str, scale: float):
        self.symbol: str = symbol  #: Symbol of this unit.
        #: Scale factor to convert this unit to another unit of the same type.
        self.scale: float = scale

    def __str__(self):
        return self.symbol

    def convert_to(self, value, target_unit: str | typing.Self):
        """
        Convert this unit to another unit.

        :param value: The value to be converted.
        :param target_unit: Target unit.
        :type target_unit: str or Unit
        :return: Converted value.
        """
        if not isinstance(target_unit, Unit):
            target_unit = self.from_str(target_unit)
        if target_unit is self:
            return value
        return value * (self.scale / target_unit.scale)

    @classmethod
    def type(cls) -> str:
        """
        Type of represented units.

        :return: Type name.
        :rtype: str
        """
        return cls.__name__.lower()

    @classmethod
    @cache
    def allowed_names(cls) -> list[str]:
        """
        List of all allowed names of this unit type.

        :return: Allowed names.
        :rtype: list[str]
        """
        return [member.name for member in cls] + [member.symbol for member in cls]

    @classmethod
    def from_str(cls, name: str) -> typing.Self:
        """
        Retrieve unit by name.

        :param str name: Name of unit.
        :return: Unit.
        :rtype: Unit
        """
        for member in cls:
            if name.upper() == member.name or name == member.symbol:
                return member
        raise ValueError(f"Unknown {cls.type()} unit: {name}")

    @classmethod
    def convert(cls, value, from_: str | typing.Self, to: str | typing.Self):
        """
        Convert a quantity with given unit ``from_`` to that with unit ``to``.
        These two units must be of the same type.

        :param value: The quantity to be converted.
        :param from_: Original unit of ``value``.
        :param to: Target unit of ``value``.
        :return: Converted quantity.
        """
        if not isinstance(from_, Unit):
            from_ = cls.from_str(from_)
        return from_.convert_to(value, to)

    @classmethod
    def units(cls) -> list[str]:
        """
        List of :attr:`.symbol` s of all units of this type.

        :return: List of symbols.
        :rtype: list[str]
        """
        return [member.symbol for member in cls]

    @classmethod
    def default(cls, default_unit: str | typing.Self = None) -> typing.Self:
        """
        Get global default unit of this type if ``default_unit`` is not given,
        otherwise set global default unit of this type to ``default_unit``.

        :param default_unit: Default unit to be set. Default: do not set.
        :return: Current global default unit.
        """
        if cls == Unit:
            raise TypeError(f'Cannot call {cls.default.__name__} on base class {cls.__name__}')
        if default_unit is not None:
            if not isinstance(default_unit, Unit):
                default_unit = cls.from_str(default_unit)
            _default_units[cls.type()] = default_unit
        return _default_units[cls.type()]

    @classmethod
    def as_default(cls, value, from_unit: str | typing.Self):
        """
        Convert a quantity with given unit ``from_unit`` to that with default unit of this type.

        :param value: The quantity to be converted.
        :param from_unit: Original unit of ``value``.
        :return: Converted quantity.
        """
        return cls.convert(value, from_unit, _default_units[cls.type()])

    @classmethod
    def default_to(cls, value, to_unit: str | typing.Self):
        """
        Convert a quantity with default unit of this type to that with unit ``to_unit``.

        :param value: The quantity to be converted.
        :param to_unit: Target unit of ``value``.
        :return: Converted quantity.
        """
        return cls.convert(value, _default_units[cls.type()], to_unit)

    @classmethod
    def fmt(cls, value: int | float, unit: str | typing.Self = None) -> str:
        """
        Format a value with unit.

        .. testsetup::

            import dnois
            dnois.Length.default('m')

        >>> from dnois import Length
        >>> Length.fmt(1)
        1m
        >>> Length.fmt(1, 'cm')
        1cm
        >>> Length.fmt(float('inf'))
        inf

        :param value: The value to be formatted.
        :param unit: Unit of ``value``. Default: use global default unit.
        :return: Formatted string.
        """
        if value == float('inf') or value == float('-inf'):
            return str(value)
        if value == float('nan'):
            return 'nan'
        if unit is None:
            unit = cls.default()
        if not isinstance(unit, Unit):
            unit = cls.from_str(unit)
        return fmt(value) + unit.symbol


class Length(Unit):
    """Length units."""
    KILOMETER = ('km', 1e3)  #: Kilometer.
    METER = ('m', 1.0)  #: Meter.
    DECIMETER = ('dm', 1e-1)  #: Decimeter.
    CENTIMETER = ('cm', 1e-2)  #: Centimeter.
    MILLIMETER = ('mm', 1e-3)  #: Millimeter.
    MICROMETER = ('um', 1e-6)  #: Micrometer.
    NANOMETER = ('nm', 1e-9)  #: Nanometer.
    PICOMETER = ('pm', 1e-12)  #: Picometer.
    ANGSTROM = ('A', 1e-10)  #: Angstrom.
    INCH = ('inch', 2.54e-2)  #: Inch.


class Angle(Unit):
    """Angle units."""
    RADIAN = ('rad', 1.0)  #: Radian.
    MRADIAN = ('mrad', 1e-3)  #: Milliradian.
    DEGREE = ('deg', math.pi / 180)  #: Degree.
    ANGLE_MIN = ('angle_min', math.pi / 180 / 60)  #: Angular minute.
    ANGLE_SEC = ('angle_sec', math.pi / 180 / 3600)  #: Angular second.


_default_units = {
    'length': Length.METER,
    'angle': Angle.RADIAN,
}
UnitType = typing.Literal['length', 'angle']


def get_default(unit_type: UnitType) -> str:
    """
    Get global default unit.

    :param str unit_type: Unit type, either ``'length'`` or ``'angle'``.
    :return: Symbol of the global default unit.
    :rtype: str
    """
    default = _default_units.get(unit_type, None)
    if default is None:
        raise ValueError(f'Unknown unit type: {unit_type}')
    return default.symbol


def set_default(unit_type: UnitType, unit: str):
    """
    Set global default length unit.

    :param str unit_type: Unit type, either ``'length'`` or ``'angle'``.
    :param str unit: Specified global default unit.
    """
    if unit_type not in _default_units:
        raise ValueError(f'Unknown unit type: {unit_type}')
    _default_units[unit_type] = _default_units[unit_type].from_str(unit)


def convert(
    value, from_: str, to: str, unit_type: UnitType = None
):  # trailing underline required because "from" is a keyword
    """
    Convert a quantity with given unit ``from_`` to that with unit ``to``.
    These two units must be of the same type.

    :param value: The quantity to be converted.
    :param str from_: Original unit of ``value``.
    :param str to: Target unit of ``value``.
    :param str unit_type: Unit type, either ``'length'`` or ``'angle'``.
        If not given, it will be automatically determined from the symbols.
    :return: Converted quantity.
    """
    if unit_type is None:
        unit_cls = _recognize_unit_cls(from_, to)
    else:
        if unit_type in _default_units:
            unit_cls = _default_units[unit_type].__class__
        else:
            raise ValueError(f'Unknown unit type: {unit_type}')
    return unit_cls.convert(value, from_, to)


def _recognize_unit_cls(*units):
    types = [u.__class__ for u in _default_units.values()]  # list of Enums
    for t in types:
        if all(u in t.allowed_names() for u in units):
            unit_cls = t
            return unit_cls
    raise ValueError(f'Cannot recognize unit type from {units}')
