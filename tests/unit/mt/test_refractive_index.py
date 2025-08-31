import csv
import logging
from pathlib import Path
import unittest

import dnois
from dnois.mt import get

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def setup_material_data(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    data = map(
        # name,t,p,wl,n
        lambda row: (row[0], float(row[1]), float(row[2]), float(row[3]), float(row[4])),
        data
    )
    data = list(data)

    catalogs = [item[0].split(':')[0] for item in data]
    catalogs = set(catalogs)
    for catalog in catalogs:
        path = Path('resources') / 'material' / 'agf' / '2024R2' / f'{catalog}.AGF'
        if not path.exists():
            path = path.with_suffix('.agf')
        dnois.ext.zmx.load_agf(path)
        logger.info(f'Loaded glass catalog: {catalog}')

    return data


def setup_air_data(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    data = map(lambda row: map(float, row), data)  # t,p,wl,n
    data = list(data)
    return data


test_data = setup_material_data('resources/material/thermal/materials.csv')
test_air_data = setup_air_data('resources/material/thermal/air.csv')


class TestRefractiveIndex(unittest.TestCase):

    @dnois.config(temperature_affect_n=True, pressure_affect_n=True)
    def test_air_n(self):
        for t, p, wl, n in test_air_data:
            with self.subTest(t=t, p=p, wl=wl):
                wl = dnois.Length.as_default(wl, 'um')
                n_computed = get('air').n_abs(wl, t, p)
                self.assertAlmostEqual(n_computed, n, places=6)

    @dnois.config(temperature_affect_n=True, pressure_affect_n=True)
    def test_material_n_relative(self):
        for name, t, p, wl, n in test_data:
            wl = dnois.Length.as_default(wl, 'um')
            n_computed = get(name).n_rel(wl, t, p)
            with self.subTest(name=name, t=t, p=p, wl=wl):
                self.assertAlmostEqual(n_computed, n, places=6)
