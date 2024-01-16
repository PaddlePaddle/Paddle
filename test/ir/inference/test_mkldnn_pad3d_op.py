# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
from functools import partial
from typing import Dict, List

import hypothesis.strategies as st
import numpy as np
from auto_scan_test import IgnoreReasons, MkldnnAutoScanTest
from hypothesis import given
from program_config import (
    OpConfig,
    ProgramConfig,
    TensorConfig,
    create_fake_model,
    create_quant_model,
)

import paddle
import paddle.inference as paddle_infer


class TestOneDNNPad3DOp(MkldnnAutoScanTest):
    def sample_program_configs(self, *args, **kwargs):
        def generate_input(*args, **kwargs):
            return np.random.random(kwargs['in_shape']).astype(np.float32)

        def generate_paddings():
            return np.random.randint(0, 4, size=(6)).astype(np.int32)

        pad3d_op = OpConfig(
            type="pad3d",
            inputs={"X": ["input_data"], "Paddings": ["paddings_data"]},
            outputs={"Out": ["output_data"]},
            attrs={
                "mode": "constant",
                "data_format": kwargs['data_format'],
                "paddings": kwargs['paddings'],
                "use_mkldnn": True,
            },
        )

        program_config = ProgramConfig(
            ops=[pad3d_op],
            weights={},
            inputs={
                "input_data": TensorConfig(
                    data_gen=partial(generate_input, *args, **kwargs)
                ),
                "paddings_data": TensorConfig(data_gen=generate_paddings),
            },
            outputs=["output_data"],
        )

        yield program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_mkldnn=True)
        yield config, (1e-5, 1e-5)

    @given(
        data_format=st.sampled_from(['NCDHW', 'NDHWC']),
        use_paddings_tensor=st.sampled_from([True, False]),
        in_shape=st.sampled_from(
            [[2, 3, 4, 5, 6], [1, 4, 1, 3, 2], [4, 3, 2, 1, 1], [1, 1, 1, 1, 1]]
        ),
        paddings=st.sampled_from(
            [
                [0, 0, 0, 0, 0, 0],
                [1, 2, 0, 1, 2, 1],
                [2, 5, 11, 3, 4, 3],
                [0, 5, 0, 1, 0, 2],
            ]
        ),
    )
    def test(self, *args, **kwargs):
        self.run_test(quant=False, *args, **kwargs)

    def pir_run_test_config(
        self, model, params, prog_config, pred_config, feed_data
    ) -> Dict[str, np.ndarray]:
        """
        Test a single case.
        """
        paddle.set_flags({'FLAGS_enable_pir_in_executor': True})
        pred_config.set_model_buffer(model, len(model), params, len(params))
        pred_config.switch_ir_optim(False)
        pred_config.enable_new_executor()
        predictor = paddle_infer.create_predictor(pred_config)
        self.available_passes_in_framework = (
            self.available_passes_in_framework
            | set(pred_config.pass_builder().all_passes())
        )
        for name, _ in prog_config.inputs.items():
            input_tensor = predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(feed_data[name]["data"])
            if feed_data[name]["lod"] is not None:
                input_tensor.set_lod(feed_data[name]["lod"])
        predictor.run()
        result = {}
        for out_name, o_name in zip(
            prog_config.outputs, predictor.get_output_names()
        ):
            result[out_name] = predictor.get_output_handle(o_name).copy_to_cpu()
        paddle.set_flags({'FLAGS_enable_pir_in_executor': False})
        return result

    def pir_run_test(self, quant=False, *args, **kwargs):
        status = True

        for prog_config in self.sample_program_configs(*args, **kwargs):
            # if program is invalid, we should skip that cases.
            if not self.is_program_valid(prog_config):
                continue

            paddle.set_flags({'FLAGS_enable_pir_in_executor': False})
            model, params = create_fake_model(prog_config)
            if quant:
                model, params = create_quant_model(model, params)

            feed_data = {}
            for name, tensor_config in prog_config.inputs.items():
                feed_data[name] = {
                    "data": tensor_config.data,
                    "lod": tensor_config.lod,
                }
            results: List[Dict[str, np.ndarray]] = []

            # baseline: cpu no ir_optim run
            base_config = self.create_inference_config(ir_optim=False)
            results.append(
                self.pir_run_test_config(
                    model, params, prog_config, base_config, feed_data
                )
            )
            self.success_log(f"baseline program_config: {prog_config}")
            self.success_log(
                f"baseline predictor_config: {self.inference_config_str(base_config)}"
            )

            for pred_config, (atol, rtol) in self.sample_predictor_configs(
                prog_config
            ):
                # skip info
                ignore_flag = False
                for ignore_info in self.ignore_cases:
                    if ignore_info[0](prog_config, pred_config):
                        ignore_flag = True
                        if (
                            ignore_info[1]
                            == IgnoreReasons.MKLDNN_ACCURACY_ERROR
                        ):
                            self.ignore_log(
                                f"[MKLDNN_ACCURACY_ERROR] {ignore_info[2]} vs {self.inference_config_str(pred_config)}"
                            )
                        else:
                            raise NotImplementedError
                        break

                if os.path.exists(self.cache_dir):
                    shutil.rmtree(self.cache_dir)
                if not os.path.exists(self.cache_dir):
                    os.mkdir(self.cache_dir)

                try:
                    results.append(
                        self.run_test_config(
                            model, params, prog_config, pred_config, feed_data
                        )
                    )
                    self.assert_tensors_near(
                        atol, rtol, results[-1], results[0]
                    )

                    self.success_log(f"program_config: {prog_config}")
                    self.success_log(
                        f"predictor_config: {self.inference_config_str(pred_config)}"
                    )
                except Exception as e:
                    self.fail_log(f"program_config: {prog_config}")
                    self.fail_log(
                        f"predictor_config: {self.inference_config_str(pred_config)}"
                    )
                    self.fail_log(f"\033[1;31m ERROR INFO: {e}\033[0m")
                    if not ignore_flag:
                        status = False
                    continue

        self.assertTrue(status)

    @given(
        data_format=st.sampled_from(['NCDHW', 'NDHWC']),
        use_paddings_tensor=st.sampled_from([True, False]),
        in_shape=st.sampled_from(
            [[2, 3, 4, 5, 6], [1, 4, 1, 3, 2], [4, 3, 2, 1, 1], [1, 1, 1, 1, 1]]
        ),
        paddings=st.sampled_from(
            [
                [0, 0, 0, 0, 0, 0],
                [1, 2, 0, 1, 2, 1],
                [2, 5, 11, 3, 4, 3],
                [0, 5, 0, 1, 0, 2],
            ]
        ),
    )
    def test_pir(self, *args, **kwargs):
        self.pir_run_test(quant=False, *args, **kwargs)


if __name__ == "__main__":
    unittest.main()
