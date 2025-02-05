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
import time
import unittest

import numpy as np

from paddle import base
from paddle.cinn.common import DefaultHostTarget, DefaultNVGPUTarget
from paddle.cinn.frontend import Interpreter

enable_gpu = sys.argv.pop()
model_dir = sys.argv.pop()
print("enable_gpu is : ", enable_gpu)
print("model_dir is : ", model_dir)


class TestLoadResnetModel(unittest.TestCase):
    def setUp(self):
        if enable_gpu == "ON":
            self.target = DefaultNVGPUTarget()
        else:
            self.target = DefaultHostTarget()
        self.model_dir = model_dir
        self.x_shape = [1, 3, 224, 224]
        self.target_tensor = 'save_infer_model/scale_0'
        self.input_tensor = 'image'

    def get_paddle_inference_result(self, model_dir, data):
        config = base.core.AnalysisConfig(
            model_dir + '/__model__', model_dir + '/params'
        )
        config.disable_gpu()
        config.switch_ir_optim(False)
        self.paddle_predictor = base.core.create_paddle_predictor(config)
        data = base.core.PaddleTensor(data)
        results = self.paddle_predictor.run([data])
        get_tensor = self.paddle_predictor.get_output_tensor(
            self.target_tensor
        ).copy_to_cpu()
        return get_tensor

    def apply_test(self):
        start = time.time()
        x_data = np.random.random(self.x_shape).astype("float32")
        self.executor = Interpreter([self.input_tensor], [self.x_shape])
        print("self.mode_dir is:", self.model_dir)
        # True means load combined model
        self.executor.load_paddle_model(
            self.model_dir, self.target, True, "resnet18"
        )
        end1 = time.time()
        print("load_paddle_model time is: %.3f sec" % (end1 - start))
        a_t = self.executor.get_tensor(self.input_tensor)
        a_t.from_numpy(x_data, self.target)
        out = self.executor.get_tensor(self.target_tensor)
        out.from_numpy(np.zeros(out.shape(), dtype='float32'), self.target)
        for i in range(10):
            self.executor.run()

        repeat = 10
        end4 = time.perf_counter()
        for i in range(repeat):
            self.executor.run()
        end5 = time.perf_counter()
        print(
            f"Repeat {repeat} times, average Executor.run() time is: {(end5 - end4) * 1000 / repeat:.3f} ms"
        )

        a_t.from_numpy(x_data, self.target)
        out.from_numpy(np.zeros(out.shape(), dtype='float32'), self.target)
        self.executor.run()

        out = out.numpy(self.target)
        target_result = self.get_paddle_inference_result(self.model_dir, x_data)

        print("result in test_model: \n")
        out = out.reshape(-1)
        target_result = target_result.reshape(-1)
        for i in range(0, min(out.shape[0], 200)):
            if np.abs(out[i] - target_result[i]) > 1.0:
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
        # TODO(thisjiang): revert atol to 1e-3 after fix inference mul problem
        np.testing.assert_allclose(out, target_result, atol=1.0)

    def test_model(self):
        self.apply_test()


if __name__ == "__main__":
    unittest.main()
