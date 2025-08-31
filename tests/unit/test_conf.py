import unittest

import dnois
from dnois.conf import config


class TestConfig(unittest.TestCase):
    def test_init_with_invalid_config(self):
        with self.assertRaises(ValueError) as context:
            config(invalid_config='value')
        self.assertIn("Unknown config item: invalid_config", str(context.exception))

    def test_context_manager_basic(self):
        original = dnois.conf.float_print_fmt
        with config(float_print_fmt='.3f'):
            self.assertEqual(dnois.conf.float_print_fmt, '.3f')
        self.assertEqual(dnois.conf.float_print_fmt, original)

    def test_context_manager_multiple_configs(self):
        original1 = dnois.conf.float_print_fmt
        original2 = dnois.conf.detection_radius_eps

        with config(float_print_fmt='.2f', detection_radius_eps=1e-3):
            self.assertEqual(dnois.conf.float_print_fmt, '.2f')
            self.assertEqual(dnois.conf.detection_radius_eps, 1e-3)

        self.assertEqual(dnois.conf.float_print_fmt, original1)
        self.assertEqual(dnois.conf.detection_radius_eps, original2)

    def test_context_manager_exception_handling(self):
        original_fmt = dnois.conf.float_print_fmt
        try:
            with config(float_print_fmt='.1f'):
                self.assertEqual(dnois.conf.float_print_fmt, '.1f')
                raise ValueError("Test exception")
        except ValueError:
            pass
        self.assertEqual(dnois.conf.float_print_fmt, original_fmt)

    def test_decorator_usage(self):
        @config(float_print_fmt='.4f')
        def test_function():
            return dnois.conf.float_print_fmt

        original_fmt = dnois.conf.float_print_fmt
        result = test_function()
        self.assertEqual(result, '.4f')
        self.assertEqual(dnois.conf.float_print_fmt, original_fmt)

    def test_nested_context_managers(self):
        original_fmt = dnois.conf.float_print_fmt
        with config(float_print_fmt='.3f'):
            self.assertEqual(dnois.conf.float_print_fmt, '.3f')
            with config(float_print_fmt='.1f'):
                self.assertEqual(dnois.conf.float_print_fmt, '.1f')
            self.assertEqual(dnois.conf.float_print_fmt, '.3f')
        self.assertEqual(dnois.conf.float_print_fmt, original_fmt)
