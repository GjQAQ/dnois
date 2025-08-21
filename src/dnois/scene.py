import torch

from dnois.base import ShapeError
from dnois.base.typing import Ts, Any
from dnois.isp import srgb2linear

__all__ = [
    'ImageScene',
    'PointCloudScene',
    'Scene',
    'ViewArrayScene',
]


class Scene:  # not used at present, just a dummy base class
    pass


def _ts_info(ts: Ts) -> dict[str, Any]:
    return {
        'shape': ts.shape,
        'max': ts.max().item(),
        'min': ts.min().item(),
        'dtype': ts.dtype,
        'device': ts.device,
    }


class ImageScene(Scene):
    """
    Class for scenes represented by its pinhole image, i.e. perspective projection
    to the image plane via a pinhole camera.

    Given the intrinsic parameters of a pinhole camera and corresponding depth map,
    the image can be projected backward to reconstruct a point cloud.
    """

    def __init__(
        self,
        image: Ts,  # (B x )(P x )C x H x W
        depth: Ts = None,  # (B x )H x W
        intrinsic: Ts = None,  # (B|1 x )3 x 3
        polarized: bool = False,
        batched: bool = True
    ):
        i_ndim = 3 + bool(polarized) + bool(batched)
        i_shape = f'({"B, " if batched else ""}{"P, " if polarized else ""}C, H, W)'
        if image.ndim != i_ndim:
            raise ShapeError(f'Image with shape {i_shape} expected, got {image.shape}')
        if depth is not None:
            d_ndim = 2 + bool(batched)
            d_shape = f'({"B, " if batched else ""}H, W)'
            if depth.ndim != d_ndim:
                raise ShapeError(f'Depth map with shape {d_shape} expected, got {depth.shape}')
            if image.shape[-2:] != depth.shape[-2:]:
                raise ShapeError(f'The spatial dimensions of image ({image.shape[-2:]}) '
                                 f'and depth map ({depth.shape[-2:]}) do not match')
            if batched and image.size(0) != depth.size(0):
                raise ShapeError(f'Batch sizes of image ({image.size(0)}) and depth map '
                                 f'({depth.size(0)}) are different')
            if depth.le(0).any():
                raise ValueError(f'Depth must be positive')
            if depth.isnan().any():
                raise ValueError(f'NaN detected in depth map')
        if intrinsic is not None:
            if intrinsic.shape[-2:] != (3, 3):
                raise ShapeError(f'The last two dimensions of intrinsic must be (3, 3)')
            if batched:
                if intrinsic.ndim != 3:
                    raise ShapeError(f'Intrinsic matrix of shape (B, 3, 3) expected, got{intrinsic.shape}')
                if intrinsic.size(0) not in (image.size(0), 1):
                    raise ShapeError(f'Batch size of intrinsic matrix must be equal to '
                                     f'that of the image ({image.size(0)}) or 1, got {intrinsic.shape}')
            elif intrinsic.ndim != 2:
                raise ShapeError(f'Intrinsic matrix of shape (3, 3) expected, got {intrinsic.shape}')

        self._image = image
        self._depth = depth
        self._intrinsic = intrinsic
        self._polarized = polarized
        self._batched = batched

    def __repr__(self):
        iif_str = ', '.join(f'{k}={v}' for k, v in _ts_info(self._image).items())
        iif_str = f'image_info=({iif_str})'
        if self._depth is None:
            dif_str = 'depth=None'
        else:
            dif_str = ', '.join(f'{k}={v}' for k, v in _ts_info(self._depth).items())
            dif_str = f'depth_info=({dif_str})'
        description = ', '.join([
            iif_str,
            dif_str,
            f'polarized={self._polarized}',
            f'batched={self._batched}',
            f'intrinsic={self._intrinsic}',
        ])
        return f'{self.__class__.__name__}({description})'

    def batch(self) -> 'ImageScene':
        if self._batched:
            return self
        else:
            return ImageScene(
                self._image.unsqueeze(0),
                None if self._depth is None else self._depth.unsqueeze(0),
                self._intrinsic, self._polarized, True
            )

    def srgb2linear(self) -> 'ImageScene':
        return ImageScene(srgb2linear(self._image), self._depth, self._intrinsic, self._polarized, self._batched)

    @property
    def image(self) -> Ts:
        return self._image

    @property
    def depth(self) -> Ts | None:
        return self._depth

    @property
    def intrinsic(self) -> Ts | None:
        return self._intrinsic

    @property
    def batch_size(self) -> int:
        return self._image.size(0) if self._batched else 0

    @property
    def n_plr(self) -> int:
        return self._image.size(-4) if self._polarized else 0

    @property
    def n_wl(self) -> int:
        return self._image.size(-3)

    @property
    def height(self) -> int:
        return self._image.size(-2)

    @property
    def width(self) -> int:
        return self._image.size(-1)

    @property
    def depth_aware(self) -> bool:
        return self._depth is not None

    @property
    def depth_shape(self) -> torch.Size:
        if self.batch_size == 0:
            return torch.Size([self.height, self.width])
        else:
            return torch.Size([self.batch_size, self.height, self.width])


class PointCloudScene(Scene):
    """
    .. warning::

        This class is experimental.

    :param Tensor locations: Coordinates of points in :ref:`CCS <guide_imodel_cameras_coordinate_system>`.
        A tensor of shape ``(N, 3)``.
    :param Tensor luminance: Luminance of points. A tensor of shape ``(N_wl, N)``.
    """

    def __init__(self, locations: Ts, luminance: Ts):
        if not (locations.ndim == 2 and locations.size(-1) == 3):
            raise ShapeError(f'Locations of shape (N, 3) expected, got {locations.shape}')
        if not (luminance.ndim == 2 and luminance.size(1) == locations.size(0)):
            raise ShapeError(f'Luminance of shape (N_wl, {locations.size(0)}) expected, got {luminance.shape}')

        self._locations = locations
        self._luminance = luminance

    @property
    def locations(self):
        return self._locations

    @property
    def luminance(self):
        return self._luminance

    @property
    def n_wl(self):
        return self.luminance.size(0)

    @property
    def n_points(self):
        return self.luminance.size(-1)


class ViewArrayScene(Scene):

    def __init__(
        self,
        image: Ts | list[list[Ts]],  # (B x )N1 x N2 x C x H x W
        batched: bool = True,
    ):
        if not torch.is_tensor(image):
            image = torch.stack([torch.stack(img_row, -4) for img_row in image], -5)
        if batched and image.ndim != 6:
            raise ShapeError(f'image should be 6 dimensional, got {image.shape}')
        if not batched and image.ndim != 5:
            raise ShapeError(f'image should be 5 dimensional, got {image.shape}')

        self.nh = image.size(-5)
        self.nw = image.size(-4)
        self._views = [
            [ImageScene(img, batched=batched) for img in img_row.unbind(-4)]
            for img_row in image.unbind(-5)
        ]
        self._batched = batched

    def as_tensor(self) -> Ts:
        return torch.stack([
            torch.stack([s.image for s in scene_row], -4)
            for scene_row in self._views
        ], -5)

    def as_scenes(self) -> list[list[ImageScene]]:
        return [[s for s in scene_row] for scene_row in self._views]

    def batch(self) -> 'ViewArrayScene':
        if self._batched:
            return self
        # Create a new ViewArrayScene with batched ImageScene objects
        return ViewArrayScene(self.as_tensor().unsqueeze(0), batched=True)

    def srgb2linear(self) -> 'ViewArrayScene':
        return ViewArrayScene([
            [srgb2linear(s.image) for s in img_row]
            for img_row in self._views
        ], batched=self._batched)

    @property
    def images(self):
        return [
            [img_scene.image for img_scene in img_row]
            for img_row in self._views
        ]

    @property
    def batch_size(self):
        return self._views[0][0].batch_size

    @property
    def n_wl(self):
        return self._views[0][0].n_wl

    @property
    def n_vertical(self):
        return len(self._views)

    @property
    def n_horizontal(self):
        return len(self._views[0])

    @property
    def height(self):
        return self._views[0][0].height

    @property
    def width(self):
        return self._views[0][0].width
