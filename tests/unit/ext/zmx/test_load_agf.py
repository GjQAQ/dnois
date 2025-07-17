import unittest

from dnois.ext.zmx import load_agf
from dnois.mt import get


class TestLoadAGF(unittest.TestCase):
    wl = 0.58756180e-6
    materials = [
        ('BK7', 1.51680003),
        ('F5', 1.60342026),
        ('SK2', 1.60738097),
    ]

    def test_load_agf(self):
        load_agf('resources/material/agf/2024R2/SCHOTT.AGF')

        for material, n in self.materials:
            with self.subTest(material=material):
                self.assertAlmostEqual(get(material).n(self.wl), n, places=8)
