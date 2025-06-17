import unittest

import torch

import dnois


class TestZernike(unittest.TestCase):
    def test_zernike(self):
        grid_size = 100
        # single precision fails when k=11
        x = torch.linspace(-1, 1, grid_size, dtype=torch.double)
        x, y = torch.meshgrid(x, x, indexing='xy')
        r = torch.sqrt(x ** 2 + y ** 2)
        theta = torch.atan2(y, x)

        for k in range(1, 12):
            zernike = dnois.zernike(r, theta, k)
            zernike_gt = self.zernike_gt(r, theta, k)
            self.assertTrue(torch.allclose(zernike, zernike_gt))

    @staticmethod
    def zernike_gt(r, theta, k):
        if k == 1:
            return torch.ones_like(r)
        elif k == 2:
            return 2 * r * theta.cos()
        elif k == 3:
            return 2 * r * theta.sin()
        elif k == 4:
            return 3 ** 0.5 * (2 * r ** 2 - 1)
        elif k == 5:
            return 6 ** 0.5 * r ** 2 * torch.sin(2 * theta)
        elif k == 6:
            return 6 ** 0.5 * r ** 2 * torch.cos(2 * theta)
        elif k == 7:
            return 8 ** 0.5 * (3 * r ** 3 - 2 * r) * theta.sin()
        elif k == 8:
            return 8 ** 0.5 * (3 * r ** 3 - 2 * r) * theta.cos()
        elif k == 9:
            return 8 ** 0.5 * r ** 3 * torch.sin(3 * theta)
        elif k == 10:
            return 8 ** 0.5 * r ** 3 * torch.cos(3 * theta)
        elif k == 11:
            return 5 ** 0.5 * (6 * r ** 4 - 6 * r ** 2 + 1)
        else:
            raise NotImplementedError()
