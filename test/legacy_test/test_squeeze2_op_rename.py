#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from test_attribute_var import UnittestBase

import paddle

paddle.enable_static()

from paddle.framework import in_pir_mode


class TestSqueeze2AxesTensor(UnittestBase):
    def init_info(self):
        self.shapes = [[2, 3, 4]]
        self.save_path = os.path.join(self.temp_dir.name, 'squeeze_tensor')

    def test_static(self):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):
            fc = paddle.nn.Linear(4, 10)
            x = paddle.randn([2, 3, 4])
            x.stop_gradient = False
            feat = fc(x)  # [2,3,10]
            feat = paddle.unsqueeze(feat, [0, 2])  # [1, 2, 3, 1, 10]
            # axes is a Variable
            axes = paddle.assign([0, 2])
            out = paddle.squeeze(feat, axes)
            out2 = paddle.squeeze(feat, axes)

            sgd = paddle.optimizer.SGD()
            sgd.minimize(paddle.mean(out))
            if not in_pir_mode():
                self.assertTrue("Var[" in str(main_prog))

            exe = paddle.static.Executor()
            exe.run(startup_prog)
            res = exe.run(fetch_list=[feat, out, out2])
            self.assertEqual(res[0].shape, (1, 2, 1, 3, 10))
            self.assertEqual(res[1].shape, (2, 3, 10))
            self.assertEqual(res[2].shape, (2, 3, 10))

            paddle.static.save_inference_model(self.save_path, [x], [out], exe)
            # Test for Inference Predictor
            infer_out = self.infer_prog()
            self.assertEqual(infer_out.shape, (2, 3, 10))


class TestSqueeze2AxesTensorList(UnittestBase):
    def init_info(self):
        self.shapes = [[2, 3, 4]]
        self.save_path = os.path.join(self.temp_dir.name, 'squeeze_tensor')

    def test_static(self):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):
            fc = paddle.nn.Linear(4, 10)
            x = paddle.randn([2, 3, 4])
            x.stop_gradient = False
            feat = fc(x)  # [2,3,10]
            feat = paddle.unsqueeze(feat, [0, 2])  # [1, 2, 3, 1, 10]
            # axes is a list[Variable]
            axes = [
                paddle.full([1], 0, dtype='int32'),
                paddle.full([1], 2, dtype='int32'),
            ]
            out = paddle.squeeze(feat, axes)
            out2 = paddle.squeeze(feat, axes)

            sgd = paddle.optimizer.SGD()
            sgd.minimize(paddle.mean(out))
            if not in_pir_mode():
                self.assertTrue("Vars[" in str(main_prog))

            exe = paddle.static.Executor()
            exe.run(startup_prog)
            res = exe.run(fetch_list=[feat, out, out2])
            self.assertEqual(res[0].shape, (1, 2, 1, 3, 10))
            self.assertEqual(res[1].shape, (2, 3, 10))
            self.assertEqual(res[2].shape, (2, 3, 10))

            paddle.static.save_inference_model(self.save_path, [x], [out], exe)
            # Test for Inference Predictor
            infer_out = self.infer_prog()
            self.assertEqual(infer_out.shape, (2, 3, 10))


if __name__ == "__main__":
    unittest.main()
