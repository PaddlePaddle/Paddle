# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from utils import (
    extra_cc_args,
    extra_nvcc_args,
    paddle_includes,
)

import paddle
from paddle.inference import Config, create_predictor
from paddle.utils.cpp_extension import get_build_directory, load
from paddle.utils.cpp_extension.extension_utils import run_cmd

# Because Windows don't use docker, the shared lib already exists in the
# cache dir, it will not be compiled again unless the shared lib is removed.
file = f'{get_build_directory()}\\infer_custom\\infer_custom.pyd'
if os.name == 'nt' and os.path.isfile(file):
    cmd = f'del {file}'
    run_cmd(cmd, True)

# Compile and load custom op Just-In-Time.
custom_inplace = load(
    name='infer_custom',
    sources=['custom_inplace.cu'],
    extra_include_paths=paddle_includes,  # add for Coverage CI
    extra_cxx_cflags=extra_cc_args,  # test for cflags
    extra_cuda_cflags=extra_nvcc_args,  # test for cflags
    verbose=True,
)


class TestInplaceNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc = paddle.nn.Linear(4, 4)

    def forward(self, x):
        fc_out = self.fc(x)
        out = custom_inplace.custom_relu_inplace(fc_out)
        mean_out = paddle.mean(out)
        return mean_out


@unittest.skipIf(
    not paddle.is_compiled_with_cuda(), 'should compile with cuda.'
)
class TestPredictorRunWithTensor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        net = TestInplaceNet()
        model = paddle.jit.to_static(
            net,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None, 4], dtype='float32', name='x'
                ),
            ],
            full_graph=True,
        )
        paddle.jit.save(
            model,
            os.path.join(
                self.temp_dir.name, 'test_predictor_run_model/inference'
            ),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def init_predictor(self, use_pir: bool):
        config = Config(
            os.path.join(
                self.temp_dir.name,
                'test_predictor_run_model/inference.pdmodel',
            ),
            os.path.join(
                self.temp_dir.name,
                'test_predictor_run_model/inference.pdiparams',
            ),
        )
        config.enable_use_gpu(256, 0)
        config.switch_ir_optim(False)
        config.enable_new_executor()
        if use_pir:
            config.enable_new_ir()
        predictor = create_predictor(config)
        return predictor

    def get_inputs(self):
        x = np.array([[1, 2, 3, 4], [2, 3, 4, 5]]).astype(np.float32)

        x_tensor = paddle.to_tensor(x)

        return [x_tensor]

    def get_outputs(self, predictor):
        [x_tensor] = self.get_inputs()

        input_names = predictor.get_input_names()
        x_tensor.name = input_names[0]

        # disorder
        inputs = [x_tensor]
        outputs = predictor.run(inputs)

        return outputs[0]

    def test_output(self):
        pir_predictor = self.init_predictor(True)
        pir_output = self.get_outputs(pir_predictor)
        predictor = self.init_predictor(False)
        output = self.get_outputs(predictor)
        np.testing.assert_allclose(
            output.numpy().flatten(), pir_output.numpy().flatten()
        )


if __name__ == "__main__":
    unittest.main()
