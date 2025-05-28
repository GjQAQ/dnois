import json
from pathlib import Path
import warnings

import torch

from .typing import Any, Self, cast

__all__ = [
    'AsDictMixIn',
    'AsJsonMixIn',
]


class AsDictMixIn:
    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        """
        Converts ``self`` into a ``dict`` which recursively contains only primitive Python objects.

        :rtype: dict
        """
        raise NotImplementedError(f'Class {self.__class__.__name__} does not '
                                  f'implement {self.to_dict.__name__} method')

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        """
        Constructs an instance of ``cls`` from a ``dict``.

        :param dict d: A ``dict`` typically returned by :meth:`.to_dict`.
        :return: An instance of ``cls``.
        """
        cls: type
        d = cast(type[AsDictMixIn], cls)._pre_from_dict(d)
        return cls(**d)

    def _attr2dictitem(self, name: str, keep_tensor: bool = True):
        converter = getattr(self, '_todict_' + name, None)
        if converter is not None and callable(converter):
            return converter(keep_tensor)

        attr = getattr(self, name)
        if isinstance(attr, AsDictMixIn):
            return attr.to_dict(keep_tensor)

        if torch.is_tensor(attr) and not keep_tensor:
            if attr.numel() > 100:
                warnings.warn(f'Trying to convert a too large tensor (numel={attr.numel()}) to a list')
            attr = attr.tolist()

        return attr

    @classmethod
    def _pre_from_dict(cls, d: dict) -> dict:
        return d.copy()


class AsJsonMixIn(AsDictMixIn):
    def to_json(self, **kwargs) -> str:
        """
        Converts ``self`` into a JSON string.

        :keyword kwargs: Keyword arguments passed to :func:`json.dumps`.
        :rtype: str
        """
        kwargs.setdefault('indent', 2)
        return json.dumps(self.to_dict(False), **kwargs)

    def save_json(self, file, **kwargs):
        """
        Save ``self`` into a JSON file ``file``.

        :param file: The JSON file to save. Either its path (``str`` or ``pathlib.Path``)
            or a file-like object.
        :keyword kwargs: Keyword arguments passed to :func:`json.dump`.
        """
        kwargs.setdefault('indent', 2)
        if isinstance(file, str):
            file = Path(file)
        if isinstance(file, Path):
            with file.open('w', encoding='utf-8') as fp:
                json.dump(self.to_dict(False), fp, **kwargs)  # noqa
        else:
            json.dump(self.to_dict(False), file, **kwargs)

    @classmethod
    def load_json(cls, file, **kwargs) -> Self:
        """
        Constructs an instance of ``cls`` through loading JSON from a file,
        converting it to a ``dict`` and then calling :meth:`.from_dict`.

        :param file: The JSON file to load. Either its path (``str`` or ``pathlib.Path``)
            or a file-like object.
        :type file: str or ``pathlib.Path`` or file-like object
        :param kwargs: Keyword arguments passed to :func:`json.load`.
        :return: An instance of ``cls``.
        """
        if isinstance(file, str):
            file = Path(file)
        if isinstance(file, Path):
            with file.open('r', encoding='utf-8') as f:
                d = json.load(f, **kwargs)
        else:
            d = json.load(file, **kwargs)
        return cls.from_dict(d)
