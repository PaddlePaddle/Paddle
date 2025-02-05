# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import site
import sys
import unittest

import numpy as np
from utils import check_output_allclose

import paddle
from paddle.utils.cpp_extension.extension_utils import run_cmd


class GapTestNet(paddle.nn.Layer):
    def __init__(self, gap_op):
        super().__init__()
        self.test_attr1 = [1, 2, 3]
        self.test_attr2 = 1
        self.linear = paddle.nn.Linear(96, 1)
        self.conv1 = paddle.nn.Conv2D(3, 6, kernel_size=3)
        self.conv2 = paddle.nn.Conv2D(6, 3, kernel_size=3)
        self.gap = gap_op

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gap(x, self.test_attr1, self.test_attr2)
        x = paddle.flatten(x)
        x = self.linear(x)
        return x


class TestNewCustomOpSetUpInstall(unittest.TestCase):
    def setUp(self):
        # TODO(ming1753): skip window CI because run_cmd(cmd) filed
        if os.name != 'nt':
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            # compile, install the custom op egg into site-packages under background
            cmd = f'cd {cur_dir} && {sys.executable} inference_gap_setup.py install'
            run_cmd(cmd)

            site_dir = site.getsitepackages()[0]

            custom_egg_path = [
                x for x in os.listdir(site_dir) if 'gap_op_setup' in x
            ]
            assert (
                len(custom_egg_path) == 1
            ), f"Matched egg number is {len(custom_egg_path)}."
            sys.path.append(os.path.join(site_dir, custom_egg_path[0]))

            # usage: import the package directly
            import gap_op_setup

            # `custom_relu_dup` is same as `custom_relu_dup`
            self.custom_op = gap_op_setup.gap

            # config seed
            SEED = 2021
            paddle.seed(SEED)
            paddle.framework.random._manual_program_seed(SEED)

    def test_all(self):
        if paddle.is_compiled_with_cuda() and os.name != 'nt':
            self._test_static_save_and_run_inference_predictor()

    def _test_static_save_and_run_inference_predictor(self):
        np_data = np.ones((32, 3, 7, 7)).astype("float32")
        path_prefix = "custom_op_inference/inference_gap_op"
        model = GapTestNet(self.custom_op)
        x = paddle.to_tensor(np_data)
        y = model(x)
        paddle.jit.save(
            model,
            path_prefix,
            input_spec=[
                paddle.static.InputSpec(shape=[32, 3, 7, 7], dtype='float32')
            ],
        )

        from paddle.inference import Config, create_predictor

        # load inference model
        config = Config(path_prefix + ".pdmodel", path_prefix + ".pdiparams")
        config.enable_use_gpu(500, 0)
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=0,
            precision_mode=paddle.inference.PrecisionType.Float32,
            use_static=True,
            use_calib_mode=False,
        )
        config.set_trt_dynamic_shape_info(
            {"x": [32, 3, 7, 7]},
            {"x": [32, 3, 7, 7]},
            {"x": [32, 3, 7, 7]},
        )
        predictor = create_predictor(config)
        input_tensor = predictor.get_input_handle(
            predictor.get_input_names()[0]
        )
        input_tensor.reshape(np_data.shape)
        input_tensor.copy_from_cpu(np_data.copy())
        predictor.run()
        output_tensor = predictor.get_output_handle(
            predictor.get_output_names()[0]
        )
        predict_infer = output_tensor.copy_to_cpu()
        predict = y.numpy().flatten()

        predict_infer = np.array(predict_infer).flatten()
        check_output_allclose(predict, predict_infer, "predict")


if __name__ == '__main__':
    unittest.main()
