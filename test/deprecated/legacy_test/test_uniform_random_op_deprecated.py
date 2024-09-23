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
import unittest

import numpy as np
from test_attribute_var_deprecated import UnittestBase

import paddle
from paddle.framework import in_pir_mode


class TestUniformMinMaxTensor(UnittestBase):
    def init_info(self):
        self.shapes = [[2, 3, 4]]
        self.save_path = os.path.join(self.temp_dir.name, self.path_prefix())

    def test_static(self):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):
            fc = paddle.nn.Linear(4, 10)
            x = paddle.randn([2, 3, 4])
            x.stop_gradient = False
            feat = fc(x)  # [2,3,10]
            min_v = paddle.to_tensor([0.1])
            max_v = paddle.to_tensor([0.9])
            y = paddle.uniform([2, 3, 10], min=min_v, max=max_v)
            z = paddle.uniform([2, 3, 10], min=min_v, max=max_v)

            out = feat + y + z

            sgd = paddle.optimizer.SGD()
            sgd.minimize(paddle.mean(out))
            if not in_pir_mode():
                self.assertTrue(self.var_prefix() in str(main_prog))

            exe = paddle.static.Executor()
            exe.run(startup_prog)
            res = exe.run(fetch_list=[out])
            np.testing.assert_array_equal(res[0].shape, [2, 3, 10])

            paddle.static.save_inference_model(self.save_path, [x], [out], exe)
            # Test for Inference Predictor
            infer_out = self.infer_prog()
            np.testing.assert_array_equal(res[0].shape, [2, 3, 10])

    def path_prefix(self):
        return 'uniform_random'

    def var_prefix(self):
        return "Var["


if __name__ == "__main__":
    unittest.main()
