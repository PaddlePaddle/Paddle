# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved
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

os.environ['FLAGS_enable_pir_api'] = '1'
import unittest

import numpy as np

import paddle
from paddle.inference import Config, PrecisionType, create_predictor
from paddle.jit import to_static
from paddle.static import InputSpec
from paddle.vision.models import alexnet


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.temp_dir = "./"
        net = alexnet(True)
        with paddle.pir_utils.DygraphPirGuard():
            model = to_static(
                net,
                input_spec=[InputSpec(shape=[None, 3, 224, 224], name='x')],
                full_graph=True,
            )
            paddle.jit.save(
                model, os.path.join(self.temp_dir, 'alexnet/inference')
            )

    def tearDown(self):
        # self.temp_dir.cleanup()
        pass

    def get_baseline(self):
        predictor = self.init_predictor(save_optimized_model=True)

        inputs = [
            paddle.to_tensor(0.1 * np.ones([1, 3, 224, 224]).astype(np.float32))
        ]

        outputs = predictor.run(inputs)

        return outputs[0]

    def get_test_output(self):
        predictor = self.init_predictor(save_optimized_model=False)

        inputs = [
            paddle.to_tensor(0.1 * np.ones([1, 3, 224, 224]).astype(np.float32))
        ]

        outputs = predictor.run(inputs)

        return outputs[0]

    def test_output(self):
        if paddle.is_compiled_with_cuda():
            baseline = self.get_baseline()
            test_output = self.get_test_output()

            np.testing.assert_allclose(
                baseline.numpy().flatten(),
                test_output.numpy().flatten(),
            )

    def init_predictor(self, save_optimized_model: bool):
        if save_optimized_model is True:
            config = Config(os.path.join(self.temp_dir, 'alexnet'), 'inference')
            config.enable_use_gpu(256, 0, PrecisionType.Half)
            config.enable_memory_optim()
            config.switch_ir_optim(True)
            config.set_optim_cache_dir(os.path.join(self.temp_dir, 'alexnet'))
            config.enable_save_optim_model(True)
        else:
            config = Config(
                os.path.join(self.temp_dir, 'alexnet'),
                os.path.join('_optimized'),
            )
            config.enable_use_gpu(256, 0, PrecisionType.Half)
            config.enable_memory_optim()
            config.switch_ir_optim(False)
        predictor = create_predictor(config)
        return predictor


if __name__ == '__main__':
    unittest.main()
