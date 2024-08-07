#!/usr/bin/env python3

# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

import sys
import unittest

import numpy as np

from paddle import base
from paddle.cinn.common import DefaultHostTarget, DefaultNVGPUTarget
from paddle.cinn.frontend import Interpreter

enable_gpu = sys.argv.pop()
model_dir = sys.argv.pop()


class TestLoadResnetModel(unittest.TestCase):
    def setUp(self):
        if enable_gpu == "ON":
            self.target = DefaultNVGPUTarget()
        else:
            self.target = DefaultHostTarget()

        self.model_dir = model_dir

        self.x_shape = [1, 160, 7, 7]

    def get_paddle_inference_result(self, data):
        config = base.core.AnalysisConfig(
            self.model_dir + ".pdmodel", self.model_dir + ".pdiparams"
        )
        config.disable_gpu()
        config.switch_ir_optim(False)
        self.paddle_predictor = base.core.create_paddle_predictor(config)
        data = base.core.PaddleTensor(data)
        results = self.paddle_predictor.run([data])
        return results[0].as_ndarray()

    def apply_test(self):
        np.random.seed(0)
        x_data = np.random.random(self.x_shape).astype("float32")
        self.executor = Interpreter(["resnet_input"], [self.x_shape])
        self.executor.load_paddle_model(self.model_dir, self.target, True)
        a_t = self.executor.get_tensor("resnet_input")
        a_t.from_numpy(x_data, self.target)

        out = self.executor.get_tensor("save_infer_model/scale_0.tmp_0")
        out.from_numpy(np.zeros(out.shape(), dtype='float32'), self.target)

        self.executor.run()

        out = out.numpy(self.target)
        target_result = self.get_paddle_inference_result(x_data)

        print("result in test_model: \n")
        out = out.reshape(-1)
        target_result = target_result.reshape(-1)
        # out.shape[0]
        for i in range(0, min(out.shape[0], 200)):
            if np.abs(out[i] - target_result[i]) > 1e-3:
                print(
                    "Error! ",
                    i,
                    "-th data has diff with target data:\n",
                    out[i],
                    " vs: ",
                    target_result[i],
                    ". Diff is: ",
                    out[i] - target_result[i],
                )
        np.testing.assert_allclose(out, target_result, atol=1e-3)

    def test_model(self):
        self.apply_test()
        # self.target.arch = Target.NVGPUArch()
        # self.apply_test()


if __name__ == "__main__":
    unittest.main()
