#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
import unittest

import numpy as np

import paddle
from paddle.base import core, executor
from paddle.distributed.io import (
    load_inference_model_distributed,
)
from paddle.static.io import load_inference_model

paddle.enable_static()


class TestLoadInferenceModelError(unittest.TestCase):
    def test_load_model_not_exist(self):
        place = core.CPUPlace()
        exe = executor.Executor(place)
        self.assertRaises(
            ValueError, load_inference_model, './test_not_exist_dir/model', exe
        )
        self.assertRaises(
            ValueError,
            load_inference_model_distributed,
            './test_not_exist_dir',
            exe,
        )


class TestPdmodelCompatibility(unittest.TestCase):
    """Test pdmodel compatibility with PIR mode auto-fallback functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.place = core.CPUPlace()
        self.exe = executor.Executor(self.place)
        
        # Store original PIR setting and ensure PIR is enabled for testing fallback
        self.original_pir_flag = paddle.base.framework.get_flags("FLAGS_enable_pir_api")["FLAGS_enable_pir_api"]
        paddle.base.set_flags({'FLAGS_enable_pir_api': True})

    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
        
        # Restore original PIR setting
        paddle.base.set_flags({'FLAGS_enable_pir_api': self.original_pir_flag})

    def _create_simple_model(self, save_format='pdmodel'):
        """Create a simple linear model for testing.

        Args:
            save_format (str): 'pdmodel' for legacy format, 'json' for PIR format

        Returns:
            str: Path prefix of the saved model
        """
        model_path = os.path.join(self.temp_dir.name, "test_model")

        # Save model in specified format
        if save_format == 'pdmodel':
            # Use OldIrGuard to save in legacy format
            with paddle.pir_utils.OldIrGuard():
                # Create simple linear regression model in legacy mode
                main_program = paddle.static.Program()
                startup_program = paddle.static.Program()

                with paddle.static.program_guard(main_program, startup_program):
                    # Define input
                    x = paddle.static.data(name='x', shape=[None, 10], dtype='float32')

                    # Use multiply + add operations to ensure ops are added to program
                    w_param = paddle.static.create_parameter(
                        shape=[10, 1],
                        dtype='float32',
                        name='weight',
                        default_initializer=paddle.nn.initializer.Normal(0.0, 1.0)
                    )
                    b_param = paddle.static.create_parameter(
                        shape=[1],
                        dtype='float32',
                        name='bias',
                        default_initializer=paddle.nn.initializer.Constant(0.0)
                    )

                    # Explicit operations to ensure they're added to the program
                    mul_result = paddle.fluid.layers.matmul(x, w_param)
                    y = paddle.fluid.layers.elementwise_add(mul_result, b_param)

                # Initialize parameters  
                self.exe.run(startup_program)

                # Save in legacy format
                paddle.static.save_inference_model(
                    path_prefix=model_path,
                    feed_vars=[x],
                    fetch_vars=[y],
                    executor=self.exe
                )
        else:  # json format (PIR mode)
            # Ensure PIR is enabled for this model creation
            old_pir_flag = paddle.base.framework.get_flags("FLAGS_enable_pir_api")["FLAGS_enable_pir_api"]
            try:
                paddle.base.set_flags({'FLAGS_enable_pir_api': True})
                
                # Create simple linear regression model in PIR mode
                main_program = paddle.static.Program()
                startup_program = paddle.static.Program()

                with paddle.static.program_guard(main_program, startup_program):
                    # Define input
                    x = paddle.static.data(name='x', shape=[None, 10], dtype='float32')

                    # Define parameters
                    w = paddle.create_parameter(
                        shape=[10, 1],
                        dtype='float32',
                        name='weight',
                        default_initializer=paddle.nn.initializer.Normal(0.0, 1.0)
                    )
                    b = paddle.create_parameter(
                        shape=[1],
                        dtype='float32',
                        name='bias',
                        default_initializer=paddle.nn.initializer.Constant(0.0)
                    )

                    # Compute output using @ operator for PIR compatibility
                    y = x @ w + b

                # Initialize parameters
                self.exe.run(startup_program)

                # Save in PIR format
                paddle.static.save_inference_model(
                    path_prefix=model_path,
                    feed_vars=[x],
                    fetch_vars=[y],
                    executor=self.exe,
                    program=main_program
                )
            finally:
                # Restore original PIR flag
                paddle.base.set_flags({'FLAGS_enable_pir_api': old_pir_flag})

        return model_path

    def test_auto_fallback_pdmodel_to_legacy(self):
        """Test automatic fallback from PIR mode to legacy mode for .pdmodel files."""
        # Create a model in legacy .pdmodel format
        model_path = self._create_simple_model(save_format='pdmodel')

        # Verify files exist
        pdmodel_file = model_path + ".pdmodel"
        pdiparams_file = model_path + ".pdiparams"
        json_file = model_path + ".json"

        self.assertTrue(os.path.exists(pdmodel_file), "pdmodel file should exist")
        self.assertTrue(os.path.exists(pdiparams_file), "pdiparams file should exist")
        self.assertFalse(os.path.exists(json_file), "json file should not exist")

        # Load model (should trigger auto-fallback)
        program, feed_names, fetch_targets = load_inference_model(
            path_prefix=model_path,
            executor=self.exe
        )

        # Verify successful loading
        self.assertIsNotNone(program, "Program should be loaded successfully")
        self.assertEqual(len(feed_names), 1, "Should have one feed variable")
        self.assertEqual(len(fetch_targets), 1, "Should have one fetch variable")
        self.assertEqual(feed_names[0], 'x', "Feed variable name should be 'x'")

        # Test inference
        test_data = np.random.random([1, 10]).astype('float32')
        results = self.exe.run(
            program,
            feed={feed_names[0]: test_data},
            fetch_list=fetch_targets
        )

        self.assertIsNotNone(results, "Inference results should not be None")
        self.assertEqual(len(results), 1, "Should have one output")
        self.assertEqual(results[0].shape, (1, 1), "Output shape should be (1, 1)")

    def test_priority_json_over_pdmodel(self):
        """Test that JSON format has priority over pdmodel when both exist."""
        # Create model in JSON format first
        model_path = self._create_simple_model(save_format='json')

        # Also create a pdmodel file manually
        pdmodel_file = model_path + ".pdmodel"
        with open(pdmodel_file, 'w') as f:
            f.write("# Dummy pdmodel content")

        json_file = model_path + ".json"

        # Both files should exist
        self.assertTrue(os.path.exists(json_file), "JSON file should exist")
        self.assertTrue(os.path.exists(pdmodel_file), "pdmodel file should exist")

        # Load model (should use JSON format, not pdmodel)
        program, feed_names, fetch_targets = load_inference_model(
            path_prefix=model_path,
            executor=self.exe
        )

        # Verify successful loading (JSON format was used)
        self.assertIsNotNone(program, "Program should be loaded successfully")
        self.assertEqual(len(feed_names), 1, "Should have one feed variable")
        self.assertEqual(len(fetch_targets), 1, "Should have one fetch variable")

    def test_kwargs_scenario_compatibility(self):
        """Test compatibility when using model_filename parameter."""
        # Create model directory structure
        model_dir = os.path.join(self.temp_dir.name, "model_dir")
        os.makedirs(model_dir, exist_ok=True)

        # Create a simple model and save to specific directory
        temp_model_path = self._create_simple_model(save_format='pdmodel')

        # Copy files to new structure
        import shutil
        shutil.copy(temp_model_path + ".pdmodel",
                   os.path.join(model_dir, "custom_model.pdmodel"))
        shutil.copy(temp_model_path + ".pdiparams",
                   os.path.join(model_dir, "custom_model.pdiparams"))

        # Load using model_filename parameter
        program, feed_names, fetch_targets = load_inference_model(
            path_prefix=model_dir,
            executor=self.exe,
            model_filename="custom_model"
        )

        # Verify successful loading
        self.assertIsNotNone(program, "Program should be loaded successfully")
        self.assertEqual(len(feed_names), 1, "Should have one feed variable")
        self.assertEqual(len(fetch_targets), 1, "Should have one fetch variable")

    def test_no_model_files_error(self):
        """Test error handling when neither JSON nor pdmodel files exist."""
        model_path = os.path.join(self.temp_dir.name, "nonexistent_model")

        # Loading should fail with appropriate error (FileNotFoundError for missing files)
        with self.assertRaises((FileNotFoundError, OSError, ValueError)):
            load_inference_model(
                path_prefix=model_path,
                executor=self.exe
            )




if __name__ == '__main__':
    unittest.main()
