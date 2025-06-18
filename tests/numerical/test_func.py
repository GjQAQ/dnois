import unittest

import torch

import dnois


class TestZernike(unittest.TestCase):
    def setUp(self):
        self.grid_size = 100
        # single precision fails when k=11
        x = torch.linspace(-1, 1, self.grid_size, dtype=torch.double)
        self.x, self.y = torch.meshgrid(x, x, indexing='xy')
        self.r = torch.sqrt(self.x ** 2 + self.y ** 2)
        self.theta = torch.atan2(self.y, self.x)

    def test_zernike1(self):
        self._test_zernike_item(1)

    def test_zernike2(self):
        self._test_zernike_item(2)

    def test_zernike3(self):
        self._test_zernike_item(3)

    def test_zernike4(self):
        self._test_zernike_item(4)

    def test_zernike5(self):
        self._test_zernike_item(5)

    def test_zernike6(self):
        self._test_zernike_item(6)

    def test_zernike7(self):
        self._test_zernike_item(7)

    def test_zernike8(self):
        self._test_zernike_item(8)

    def test_zernike9(self):
        self._test_zernike_item(9)

    def test_zernike10(self):
        self._test_zernike_item(10)

    def test_zernike11(self):
        self._test_zernike_item(11)

    def _test_zernike_item(self, k: int):
        self.assertTrue(torch.allclose(dnois.zernike(self.r, self.theta, k), self.zernike_gt(k)))
        if k > 8:
            return

        dx, dy = dnois.zernike_cpd(self.r, self.theta, k)
        dx_gt, dy_gt = self.zernike_cpd_gt(k)
        self.assertTrue(torch.allclose(dx, dx_gt))
        self.assertTrue(torch.allclose(dy, dy_gt))

    def zernike_gt(self, k):
        r, theta = self.r, self.theta
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

    def zernike_cpd_gt(self, k):
        x, y = self.x, self.y
        if k == 1:
            return torch.zeros_like(x), torch.zeros_like(y)
        elif k == 2:
            return torch.full_like(x, 2), torch.zeros_like(y)
        elif k == 3:
            return torch.zeros_like(x), torch.full_like(y, 2)
        elif k == 4:
            return 4 * 3 ** 0.5 * x, 4 * 3 ** 0.5 * y
        elif k == 5:
            return 2 * 6 ** 0.5 * y, 2 * 6 ** 0.5 * x
        elif k == 6:
            return 2 * 6 ** 0.5 * x, -2 * 6 ** 0.5 * y
        elif k == 7:
            return 12 * 2 ** 0.5 * x * y, 2 * 2 ** 0.5 * (3 * x ** 2 + 9 * y ** 2 - 2)
        elif k == 8:
            return 2 * 2 ** 0.5 * (9 * x ** 2 + 3 * y ** 2 - 2), 12 * 2 ** 0.5 * x * y
        else:
            raise NotImplementedError()
