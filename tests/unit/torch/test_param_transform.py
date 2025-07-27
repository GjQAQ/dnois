import unittest

from dnois.optics import rt
from dnois.torch import Transform, Scale, ParamTransformModule

import torch
from torch import nn


class ParamTransformModuleTest(unittest.TestCase):
    def test_set_transform(self):
        module = ParamTransformModule()
        module.test_param = nn.Parameter(torch.tensor(1.))
        module.set_transform('test_param', Transform.scale(2.))

        self.assertTrue(hasattr(module, '_param_transforms'))
        self.assertIsInstance(module._param_transforms, nn.ModuleDict)
        self.assertTrue('test_param' in module._param_transforms)
        self.assertIsInstance(module._param_transforms['test_param'], Scale)

        self.assertTrue(hasattr(module, '_latent_test_param'))
        self.assertIsInstance(module._latent_test_param, nn.Parameter)
        self.assertAlmostEqual(module._latent_test_param.item(), 0.5, 6)
        self.assertTrue(hasattr(module, 'test_param'))
        self.assertAlmostEqual(module.test_param.item(), 1, 6)

    def test_register_parameter_with_transform(self):
        """Test registering a parameter with transformation using nominal value."""
        module = ParamTransformModule()
        param = nn.Parameter(torch.tensor(10.0))
        
        module.register_parameter('test_param', param, Transform.scale(2.0))
        
        # Check that latent parameter was created
        self.assertTrue(hasattr(module, '_latent_test_param'))
        self.assertIsInstance(module._latent_test_param, nn.Parameter)
        self.assertAlmostEqual(module._latent_test_param.item(), 5.0, 6)  # 10/2
        
        # Check that nominal value is accessible
        self.assertAlmostEqual(module.test_param.item(), 10.0, 6)
        
        # Check that transform is registered
        self.assertIn('test_param', module._param_transforms)
        self.assertIsInstance(module._param_transforms['test_param'], Scale)

    def test_register_latent_parameter(self):
        """Test registering a parameter with transformation using latent value."""
        module = ParamTransformModule()
        latent_param = nn.Parameter(torch.tensor(3.0))
        
        module.register_latent_parameter('test_param', latent_param, Transform.scale(2.0))
        
        # Check that latent parameter is stored as-is
        self.assertTrue(hasattr(module, '_latent_test_param'))
        self.assertIs(module._latent_test_param, latent_param)
        
        # Check that nominal value is transformed
        self.assertAlmostEqual(module.test_param.item(), 6.0, 6)  # 3*2
        
        # Check that transform is registered
        self.assertIn('test_param', module._param_transforms)

    def test_register_parameter_without_transform(self):
        """Test registering a parameter without transformation (normal behavior)."""
        module = ParamTransformModule()
        param = nn.Parameter(torch.tensor(5.0))
        
        module.register_parameter('test_param', param)
        
        # Should behave like normal nn.Module
        self.assertTrue(hasattr(module, 'test_param'))
        self.assertIs(module.test_param, param)
        self.assertNotIn('test_param', getattr(module, '_param_transforms', {}))

    def test_register_parameter_without_inverse_transform(self):
        """Test that registering with non-invertible transform raises error."""
        module = ParamTransformModule()
        param = nn.Parameter(torch.tensor(5.0))
        
        # Create a transform without inverse
        transform = Transform(lambda x: x * 2)  # No inverse provided
        
        with self.assertRaises(ValueError):
            module.register_parameter('test_param', param, transform)

    def test_attribute_access_and_setting(self):
        """Test getting and setting transformed parameter values."""
        module = ParamTransformModule()
        module.register_latent_parameter('test_param', nn.Parameter(torch.tensor(2.0)), Transform.scale(3.0))
        
        # Test getting nominal value
        self.assertAlmostEqual(module.test_param.item(), 6.0, 6)
        
        # Test setting nominal value
        module.test_param = torch.tensor(12.0)
        self.assertAlmostEqual(module._latent_test_param.item(), 4.0, 6)  # 12/3
        self.assertAlmostEqual(module.test_param.item(), 12.0, 6)

    def test_attribute_setting_without_inverse(self):
        """Test that setting value without inverse transform raises error."""
        module = ParamTransformModule()
        transform = Transform(lambda x: x * 2)  # No inverse
        module.register_latent_parameter('test_param', nn.Parameter(torch.tensor(1.0)), transform)
        
        with self.assertRaises(RuntimeError):
            module.test_param = torch.tensor(4.0)

    def test_attribute_deletion(self):
        """Test deleting transformed parameters."""
        module = ParamTransformModule()
        module.register_latent_parameter('test_param', nn.Parameter(torch.tensor(1.0)), Transform.scale(2.0))
        
        # Verify parameter exists
        self.assertTrue(hasattr(module, 'test_param'))
        self.assertTrue(hasattr(module, '_latent_test_param'))
        self.assertIn('test_param', module._param_transforms)
        
        # Delete the parameter
        delattr(module, 'test_param')
        
        # Verify it's gone
        self.assertFalse(hasattr(module, 'test_param'))
        self.assertFalse(hasattr(module, '_latent_test_param'))
        self.assertNotIn('test_param', module._param_transforms)

    def test_remove_transform(self):
        """Test removing transformation from a parameter."""
        module = ParamTransformModule()
        module.register_latent_parameter('test_param', nn.Parameter(torch.tensor(2.0)), Transform.scale(3.0))
        
        # Verify transformation is active
        self.assertAlmostEqual(module.test_param.item(), 6.0, 6)
        
        # Remove transformation
        module.remove_transform('test_param')
        
        # Verify parameter is now normal (not transformed)
        self.assertAlmostEqual(module.test_param.item(), 6.0, 6)  # Keeps the nominal value
        self.assertFalse(hasattr(module, '_latent_test_param'))
        self.assertNotIn('test_param', module._param_transforms)

    def test_remove_transform_nonexistent(self):
        """Test removing transformation from non-existent parameter."""
        module = ParamTransformModule()
        
        with self.assertRaises(RuntimeError):
            module.remove_transform('nonexistent_param')

    def test_remove_transform_no_transforms(self):
        """Test removing transformation when no transforms are set."""
        module = ParamTransformModule()
        
        with self.assertRaises(RuntimeError):
            module.remove_transform('test_param')

    def test_nominal_values_property(self):
        """Test the nominal_values property."""
        module = ParamTransformModule()
        
        # Add normal parameter
        module.register_parameter('normal_param', nn.Parameter(torch.tensor(5.0)))
        
        # Add transformed parameter
        module.register_latent_parameter('transformed_param', nn.Parameter(torch.tensor(2.0)), Transform.scale(3.0))
        
        nominal_values = module.nominal_values
        
        self.assertIn('normal_param', nominal_values)
        self.assertIn('transformed_param', nominal_values)
        self.assertAlmostEqual(nominal_values['normal_param'].item(), 5.0, 6)
        self.assertAlmostEqual(nominal_values['transformed_param'].item(), 6.0, 6)
        
        # Should not contain latent parameters
        self.assertNotIn('_latent_transformed_param', nominal_values)

    def test_transformed_parameters_property(self):
        """Test the transformed_parameters property."""
        module = ParamTransformModule()
        
        # Add normal parameter
        module.register_parameter('normal_param', nn.Parameter(torch.tensor(5.0)))
        
        # Add transformed parameter
        latent_param = nn.Parameter(torch.tensor(2.0))
        transform = Transform.scale(3.0)
        module.register_latent_parameter('transformed_param', latent_param, transform)
        
        transformed_params = module.transformed_parameters
        
        self.assertIn('transformed_param', transformed_params)
        self.assertNotIn('normal_param', transformed_params)
        
        latent_param_retrieved, transform_retrieved = transformed_params['transformed_param']
        self.assertIs(latent_param_retrieved, latent_param)
        self.assertIs(transform_retrieved, transform)

    def test_different_transform_types(self):
        """Test different types of transformations."""
        module = ParamTransformModule()
        
        # Test Range transform
        module.register_latent_parameter('range_param', nn.Parameter(torch.tensor(0.0)), Transform.range(1.0, 5.0))
        self.assertAlmostEqual(module.range_param.item(), 3.0, 1)  # sigmoid(0) = 0.5, so 1 + 4*0.5 = 3
        
        # Test Gt transform
        module.register_latent_parameter('gt_param', nn.Parameter(torch.tensor(1.0)), Transform.gt(2.0))
        self.assertAlmostEqual(module.gt_param.item(), 2.0 + torch.exp(torch.tensor(1.0)).item(), 6)
        
        # Test Lt transform
        module.register_latent_parameter('lt_param', nn.Parameter(torch.tensor(1.0)), Transform.lt(10.0))
        self.assertAlmostEqual(module.lt_param.item(), 10.0 - torch.exp(torch.tensor(1.0)).item(), 6)

    def test_composite_transform(self):
        """Test composite transformations."""
        module = ParamTransformModule()
        
        # Create composite transform: scale by 2, then range [1, 5]
        composite = Transform.composite(Transform.scale(2.0), Transform.range(1.0, 5.0))
        module.register_latent_parameter('composite_param', nn.Parameter(torch.tensor(0.0)), composite)
        
        # Test the transformation
        nominal_value = module.composite_param.item()
        self.assertGreater(nominal_value, 1.0)
        self.assertLess(nominal_value, 5.0)

    def test_parameter_conversion_existing_parameter(self):
        """Test converting existing parameter to transformed parameter."""
        module = ParamTransformModule()
        
        # First register a normal parameter
        module.register_parameter('test_param', nn.Parameter(torch.tensor(10.0)))
        self.assertAlmostEqual(module.test_param.item(), 10.0, 6)
        
        # Convert to transformed parameter
        module.set_transform('test_param', Transform.scale(2.0))
        
        # Should now be transformed
        self.assertAlmostEqual(module.test_param.item(), 10.0, 6)  # Nominal value preserved
        self.assertAlmostEqual(module._latent_test_param.item(), 5.0, 6)  # Latent value is 10/2
        self.assertIn('test_param', module._param_transforms)

    def test_parameter_conversion_existing_latent_parameter(self):
        """Test converting existing latent parameter to transformed parameter."""
        module = ParamTransformModule()
        
        # First register a latent parameter
        module.register_latent_parameter('test_param', nn.Parameter(torch.tensor(3.0)), Transform.scale(2.0))
        self.assertAlmostEqual(module.test_param.item(), 6.0, 6)
        
        # Convert to different transform
        module.set_transform('test_param', Transform.scale(3.0))
        
        # Should now use new transform
        self.assertAlmostEqual(module.test_param.item(), 6.0, 6)  # Nominal value preserved
        self.assertAlmostEqual(module._latent_test_param.item(), 2.0, 6)  # Latent value is 6/3
        self.assertIn('test_param', module._param_transforms)

    def test_error_non_parameter_attribute(self):
        """Test error when trying to convert non-parameter attribute."""
        module = ParamTransformModule()
        
        # Add a non-parameter attribute
        module.some_attribute = 42
        
        with self.assertRaises(AttributeError):
            module.set_transform('some_attribute', Transform.scale(2.0))

    def test_latent_name_helpers(self):
        """Test the static helper methods for latent name conversion."""
        # Test _latent_name
        latent_name = ParamTransformModule._latent_name('test_param')
        self.assertEqual(latent_name, '_latent_test_param')
        
        # Test _nominal_name
        nominal_name = ParamTransformModule._nominal_name('_latent_test_param')
        self.assertEqual(nominal_name, 'test_param')

    def test_multiple_transformed_parameters(self):
        """Test managing multiple transformed parameters."""
        module = ParamTransformModule()
        
        # Register multiple transformed parameters
        module.register_latent_parameter('param1', nn.Parameter(torch.tensor(1.0)), Transform.scale(2.0))
        module.register_latent_parameter('param2', nn.Parameter(torch.tensor(2.0)), Transform.scale(3.0))
        module.register_latent_parameter('param3', nn.Parameter(torch.tensor(3.0)), Transform.range(0.0, 10.0))
        
        # Test all parameters
        self.assertAlmostEqual(module.param1.item(), 2.0, 6)
        self.assertAlmostEqual(module.param2.item(), 6.0, 6)
        self.assertGreater(module.param3.item(), 0.0)
        self.assertLess(module.param3.item(), 10.0)
        
        # Test nominal_values
        nominal_values = module.nominal_values
        self.assertIn('param1', nominal_values)
        self.assertIn('param2', nominal_values)
        self.assertIn('param3', nominal_values)
        
        # Test transformed_parameters
        transformed_params = module.transformed_parameters
        self.assertEqual(len(transformed_params), 3)
        self.assertIn('param1', transformed_params)
        self.assertIn('param2', transformed_params)
        self.assertIn('param3', transformed_params)
