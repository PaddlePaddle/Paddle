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
import shutil
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
    """Test pdmodel compatibility with PIR mode fallback."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.place = core.CPUPlace()
        self.exe = executor.Executor(self.place)

        self.original_pir_flag = paddle.base.framework.get_flags(
            "FLAGS_enable_pir_api"
        )["FLAGS_enable_pir_api"]
        paddle.base.set_flags({'FLAGS_enable_pir_api': True})

        self.original_fallback_env = os.environ.get(
            'PADDLE_ENABLE_PDMODEL_FALLBACK', ''
        )
        os.environ['PADDLE_ENABLE_PDMODEL_FALLBACK'] = '1'

        paddle.enable_static()

    def tearDown(self):
        self.temp_dir.cleanup()
        paddle.base.set_flags({'FLAGS_enable_pir_api': self.original_pir_flag})
        if self.original_fallback_env:
            os.environ['PADDLE_ENABLE_PDMODEL_FALLBACK'] = (
                self.original_fallback_env
            )
        else:
            os.environ.pop('PADDLE_ENABLE_PDMODEL_FALLBACK', None)

    def _create_model(self, save_format='pdmodel', name_suffix=''):
        """Create simple model: y = fc(x)"""
        model_path = os.path.join(
            self.temp_dir.name, f"model_{save_format}{name_suffix}"
        )

        if save_format == 'pdmodel':
            # Create program in OldIR mode for .pdmodel format
            with paddle.pir_utils.OldIrGuard():
                main_program = paddle.static.Program()
                startup_program = paddle.static.Program()

                with paddle.static.program_guard(main_program, startup_program):
                    x = paddle.static.data(
                        name='x', shape=[None, 10], dtype='float32'
                    )
                    # Use paddle.static.nn.fc for OldIR mode
                    y = paddle.static.nn.fc(
                        x,
                        size=1,
                        weight_attr=paddle.ParamAttr(
                            initializer=paddle.nn.initializer.Constant(0.5)
                        ),
                        bias_attr=paddle.ParamAttr(
                            initializer=paddle.nn.initializer.Constant(0.1)
                        ),
                    )

                # Initialize parameters
                self.exe.run(startup_program)

                # Validate program has ops
                if len(main_program.global_block().ops) == 0:
                    raise ValueError("Main program is empty!")

                # Save the model (must be done within OldIrGuard)
                paddle.static.save_inference_model(
                    path_prefix=model_path,
                    feed_vars=[x],
                    fetch_vars=[y],
                    executor=self.exe,
                    program=main_program,
                )
        else:
            # Create program in PIR mode for .json format
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()

            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data(
                    name='x', shape=[None, 10], dtype='float32'
                )
                w = paddle.create_parameter(
                    shape=[10, 1],
                    dtype='float32',
                    name='weight',
                    default_initializer=paddle.nn.initializer.Constant(0.5),
                )
                b = paddle.create_parameter(
                    shape=[1],
                    dtype='float32',
                    name='bias',
                    default_initializer=paddle.nn.initializer.Constant(0.1),
                )
                y = x @ w + b

            self.exe.run(startup_program)

            if len(main_program.global_block().ops) == 0:
                raise ValueError("Main program is empty!")

            paddle.static.save_inference_model(
                path_prefix=model_path,
                feed_vars=[x],
                fetch_vars=[y],
                executor=self.exe,
                program=main_program,
            )

        return model_path

    def _verify_loaded_model(
        self, program, feed_names, fetch_targets, expected_feed_name='x'
    ):
        """Verify loaded model by checking structure and running inference"""
        self.assertEqual(len(feed_names), 1)
        self.assertEqual(feed_names[0], expected_feed_name)
        self.assertEqual(len(fetch_targets), 1)

        # Run inference to verify model works
        x_data = np.random.rand(2, 10).astype('float32')
        results = self.exe.run(
            program,
            feed={feed_names[0]: x_data},
            fetch_list=fetch_targets,
            return_numpy=True,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].shape, (2, 1))

    def test_auto_fallback_pdmodel_to_legacy(self):
        """Test auto fallback from PIR to legacy mode when loading .pdmodel"""
        pdmodel_path = self._create_model('pdmodel')

        # Verify pdmodel files exist and json files don't
        self.assertTrue(
            os.path.exists(pdmodel_path + ".pdmodel"),
            "pdmodel file should exist",
        )
        self.assertTrue(
            os.path.exists(pdmodel_path + ".pdiparams"),
            "pdiparams file should exist",
        )
        self.assertFalse(
            os.path.exists(pdmodel_path + ".json"),
            "json file should not exist for pdmodel format",
        )

        program, feed_names, fetch_targets = load_inference_model(
            pdmodel_path, self.exe
        )
        self._verify_loaded_model(program, feed_names, fetch_targets)

    def test_json_priority_over_pdmodel(self):
        """Test .json priority over .pdmodel when both files exist"""
        json_path = self._create_model('json', '_json')
        pdmodel_path = self._create_model('pdmodel', '_pdmodel')
        priority_path = os.path.join(self.temp_dir.name, "priority_test")

        # Create both formats at same location
        shutil.copy(json_path + ".json", priority_path + ".json")
        shutil.copy(json_path + ".pdiparams", priority_path + ".pdiparams")
        shutil.copy(pdmodel_path + ".pdmodel", priority_path + ".pdmodel")

        # Verify both file formats exist
        self.assertTrue(os.path.exists(priority_path + ".json"))
        self.assertTrue(os.path.exists(priority_path + ".pdmodel"))

        program1, feed_names1, fetch_targets1 = load_inference_model(
            priority_path, self.exe
        )
        self.assertIsInstance(program1, paddle.base.libpaddle.pir.Program)
        self._verify_loaded_model(program1, feed_names1, fetch_targets1, 'x')

        # If .json has priority, this should still work
        # If .pdmodel has priority, this would fail
        os.remove(priority_path + ".pdmodel")
        self.assertTrue(os.path.exists(priority_path + ".json"))
        self.assertFalse(os.path.exists(priority_path + ".pdmodel"))

        # Should still load successfully from .json
        program2, feed_names2, fetch_targets2 = load_inference_model(
            priority_path, self.exe
        )
        self.assertIsInstance(program2, paddle.base.libpaddle.pir.Program)
        self._verify_loaded_model(program2, feed_names2, fetch_targets2, 'x')

    def test_pir_mode_loads_json_normally(self):
        """Test PIR mode loads .json format normally"""
        json_path = self._create_model('json')

        # Verify json files exist and pdmodel files don't
        self.assertTrue(
            os.path.exists(json_path + ".json"), "JSON file should exist"
        )
        self.assertTrue(
            os.path.exists(json_path + ".pdiparams"),
            "pdiparams file should exist",
        )
        self.assertFalse(
            os.path.exists(json_path + ".pdmodel"),
            "pdmodel file should not exist for json format",
        )

        program, feed_names, fetch_targets = load_inference_model(
            json_path, self.exe
        )
        # JSON should load in PIR mode
        self._verify_loaded_model(program, feed_names, fetch_targets, 'x')

    def test_pir_mode_rejects_pdmodel_without_fallback(self):
        """Test PIR mode rejects .pdmodel without fallback enabled"""
        pdmodel_path = self._create_model('pdmodel', '_no_fallback')

        # Verify pdmodel files exist and json files don't
        self.assertTrue(
            os.path.exists(pdmodel_path + ".pdmodel"),
            "pdmodel file should exist",
        )
        self.assertTrue(
            os.path.exists(pdmodel_path + ".pdiparams"),
            "pdiparams file should exist",
        )
        self.assertFalse(
            os.path.exists(pdmodel_path + ".json"),
            "json file should not exist for pdmodel format",
        )

        original = os.environ.get('PADDLE_ENABLE_PDMODEL_FALLBACK', '')
        try:
            os.environ['PADDLE_ENABLE_PDMODEL_FALLBACK'] = '0'
            with self.assertRaises(
                (RuntimeError, ValueError, FileNotFoundError)
            ) as context:
                load_inference_model(pdmodel_path, self.exe)

            # Verify error message is related to JSON/parsing/file not found
            error_message = str(context.exception).lower()
            self.assertTrue(
                "json" in error_message
                or "parse" in error_message
                or "does not exist" in error_message,
                f"Error should be JSON-related or file not found, got: {context.exception}",
            )
        finally:
            if original:
                os.environ['PADDLE_ENABLE_PDMODEL_FALLBACK'] = original
            else:
                os.environ.pop('PADDLE_ENABLE_PDMODEL_FALLBACK', None)

    def test_custom_model_filename_parameter(self):
        """Test compatibility when using custom model_filename parameter"""
        model_dir = os.path.join(self.temp_dir.name, "model_dir")
        os.makedirs(model_dir)
        temp_path = self._create_model('pdmodel', '_custom')
        for ext in [".pdmodel", ".pdiparams"]:
            shutil.copy(
                temp_path + ext,
                os.path.join(model_dir, "custom_model" + ext),
            )

        program, feed_names, fetch_targets = load_inference_model(
            model_dir, self.exe, model_filename="custom_model"
        )
        # Custom filename should work
        self._verify_loaded_model(program, feed_names, fetch_targets, 'x')

    def test_fallback_from_invalid_pdmodel_to_json(self):
        """Test that invalid .pdmodel raises error when fallback is enabled"""
        json_path = self._create_model('json', '_fallback')
        # Create an invalid .pdmodel file
        with open(json_path + ".pdmodel", 'w') as f:
            f.write("# Invalid pdmodel content")

        # When fallback is enabled and .pdmodel exists but is invalid,
        # it should raise an error rather than silently fallback
        os.environ['PADDLE_ENABLE_PDMODEL_FALLBACK'] = '1'
        try:
            with self.assertRaises((ValueError, RuntimeError)):
                load_inference_model(json_path, self.exe)
        finally:
            os.environ.pop('PADDLE_ENABLE_PDMODEL_FALLBACK', None)

    def test_no_model_files_error(self):
        """Test proper error handling when model files don't exist"""
        model_path = os.path.join(self.temp_dir.name, "nonexistent_model")

        with self.assertRaises((FileNotFoundError, OSError, ValueError)):
            load_inference_model(path_prefix=model_path, executor=self.exe)


if __name__ == '__main__':
    unittest.main()
