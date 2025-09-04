import unittest

from dnois.fourier import dconv2
import torch


class TestDConv(unittest.TestCase):
    def test_dconv2_basic(self):
        n = 10
        x = torch.zeros(n, n)
        x[n // 2, n // 2] = 1

        k = torch.zeros(n, n)
        k[n // 2, n // 2] = 1
        k[n // 2 - 1, n // 2 - 1] = 0.2
        k[[n // 2 - 1, n // 2 + 1], n // 2] = 0.5
        k[n // 2, [n // 2 - 1, n // 2 + 1]] = 0.5

        y = dconv2(x, k, out='same')

        self.assertTrue(torch.allclose(y, k, rtol=0, atol=1e-7))
