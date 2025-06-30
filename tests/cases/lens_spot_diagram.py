import logging
import unittest

import dnois
from dnois.optics import rt
import torch


class LensSpotTest(unittest.TestCase):
    use_analytical = True
    device = torch.device('cpu')

    def setUp(self):
        logging.basicConfig(
            format='[%(asctime)s](%(levelname)s)%(threadName)s/%(name)s:%(message)s',
            level=logging.INFO
        )
        logger = logging.getLogger(__name__)

        torch.set_grad_enabled(False)
        torch.set_default_dtype(torch.double)
        logger.info('Testing using double precision')

        dnois.conf.detection_radius_eps = 1e-2
        logger.info(f'Detection radius eps: {dnois.conf.detection_radius_eps}')

    def test_refractive(self):
        self._test_case(
            rt.CoaxialSurfaceSequence([
                rt.Conic(
                    5.939516708387155E-002, -8.203038802956715E-001,
                    material='BK7',
                    aperture=1.261748968479112E-002,
                    intersection_config=rt.IntersectionConfig(use_analytical=self.use_analytical),
                    d=1.200003174024448E-002
                ),
                rt.Spherical(
                    -3.594420479495210E-001,
                    material='vacuum',
                    aperture=1.245562055844115E-002,
                    intersection_config=rt.IntersectionConfig(use_analytical=self.use_analytical),
                    d=9.267049525903452E-002
                ),
            ]),
            [0, 2.5, 5],
            1.25e-2,
            0,
            0.6328e-6,
            [38.4865436e-6, 18.2430797e-6, 64.1525446e-6],
            [50.5945858e-6, 32.4408911e-6, 135.714123e-6]
        )

    def test_reflective(self):
        self._test_case(
            rt.CoaxialSurfaceSequence([
                rt.Conic(
                    -1.992156741645932E+000, -1, 'vacuum', 0.125, True,
                    rt.IntersectionConfig(use_analytical=self.use_analytical),
                    d=-9.960783704872819E-001)
            ]),
            [0, 5],
            0.125,
            0,
            0.6328e-6,
            [3.21444657e-11, 736.769254e-6],
            [4.22922952e-11, 2026.10066e-6]
        )

    def test_planar_diffractive(self):
        self._test_case(
            rt.CoaxialSurfaceSequence([
                rt.AsphericalRadialPhase(
                    phase_coef=[-8.456322343801885E+003, 3.757876911565806E+001, -1.295576930032466E+000],
                    material='BK7',
                    aperture=1.250000000000000E-002,
                    norm_radius=1.300000000000000E-002,
                    d=5.000000000000000E-003
                ),
                rt.Conic(
                    aperture=1.237506983572113E-002,
                    intersection_config=rt.IntersectionConfig(use_analytical=self.use_analytical),
                    d=9.573893049845186E-002
                )
            ]),
            [0, 5],
            1.25e-2,
            0,
            0.6328e-6,
            [18.3687765e-6, 121.510877e-6],
            [24.1003962e-6, 327.645421e-6]
        )

    def test_curved_diffractive(self):
        self._test_case(
            rt.CoaxialSurfaceSequence([
                rt.AsphericalRadialPhase(
                    -4.025189283182056E+001,
                    -4.108200251412145E+007,
                    [0, -7.636392109883529E+003, -2.585298019550675E+007, -2.170151388941415E+009],
                    phase_coef=[-8.481983557198455E+003, -1.071603422510122E+003, -6.324805687808447E+002],
                    material='BK7',
                    aperture=1.252539847680571E-002,
                    norm_radius=1.300000000000000E-002,
                    d=5.000000000000000E-003
                ),
                rt.Conic(
                    aperture=1.234705908181150E-002,
                    intersection_config=rt.IntersectionConfig(use_analytical=self.use_analytical),
                    d=9.573893049845186E-002
                )
            ]),
            [0, 5],
            1.25e-2,
            0,
            0.6328e-6,
            [0.159166014e-6, 160.074992e-6],
            [0.295141995e-6, 439.714966e-6],
        )

    def _test_case(
        self,
        sq: rt.CoaxialSurfaceSequence,
        fovs: list[float],
        entr_r: float,
        entr_z: float,
        wl: float,
        target_rms: list[float],
        target_geo_radius: list[float],
        tolerance: float = 1e-10
    ):
        o = rt.CoaxialRayTracing(sq).to(self.device, torch.double)

        point = o.fovd2obj([
            (0, fov_item) for fov_item in fovs
        ], float('inf'), True)
        spot_diagram = o.plot_spot_diagram(point, wl, width=1, entr_d=entr_r * 2, entr_z=entr_z)

        rms = spot_diagram.rms.tolist()
        geo_radius = spot_diagram.geo_radius.tolist()
        for i in range(len(target_rms)):
            self.assertAlmostEqual(rms[i], target_rms[i], delta=tolerance)
            self.assertAlmostEqual(geo_radius[i], target_geo_radius[i], delta=tolerance)
