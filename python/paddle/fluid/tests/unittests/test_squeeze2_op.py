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

import unittest

import numpy as np
import os
from op_test import OpTest
import paddle
from paddle.fluid.framework import program_guard, Program

from test_attribute_var import UnittestBase

paddle.enable_static()


# Correct: General.
class TestSqueezeOp(OpTest):

    def setUp(self):
        self.op_type = "squeeze2"
        self.python_api = paddle.squeeze
        self.python_out_sig = [
            "Out"
        ]  # python out sig is customized output signature.
        self.init_test_case()
        self.inputs = {"X": np.random.random(self.ori_shape).astype("float64")}
        self.init_attrs()
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.new_shape),
            "XShape": np.random.random(self.ori_shape).astype("float64")
        }

    def test_check_output(self):
        self.check_output(no_check_set=['XShape'], check_eager=True)

    def test_check_grad(self):
        self.check_grad(["X"], "Out", check_eager=True)

    def init_test_case(self):
        self.ori_shape = (1, 3, 1, 40)
        self.axes = (0, 2)
        self.new_shape = (3, 40)

    def init_attrs(self):
        self.attrs = {"axes": self.axes}


# Correct: There is mins axis.
class TestSqueezeOp1(TestSqueezeOp):

    def init_test_case(self):
        self.ori_shape = (1, 20, 1, 5)
        self.axes = (0, -2)
        self.new_shape = (20, 5)


# Correct: No axes input.
class TestSqueezeOp2(TestSqueezeOp):

    def init_test_case(self):
        self.ori_shape = (1, 20, 1, 5)
        self.axes = ()
        self.new_shape = (20, 5)


# Correct: Just part of axes be squeezed.
class TestSqueezeOp3(TestSqueezeOp):

    def init_test_case(self):
        self.ori_shape = (6, 1, 5, 1, 4, 1)
        self.axes = (1, -1)
        self.new_shape = (6, 5, 1, 4)


class TestSqueeze2AxesTensor(UnittestBase):

    def init_info(self):
        self.shapes = [[2, 3, 4]]
        self.save_path = os.path.join(self.temp_dir.name, 'squeeze_tensor')

    def test_static(self):
        main_prog = Program()
        starup_prog = Program()
        with program_guard(main_prog, starup_prog):
            fc = paddle.nn.Linear(4, 10)
            x = paddle.randn([2, 3, 4])
            x.stop_gradient = False
            feat = fc(x)  # [2,3,10]
            feat = paddle.unsqueeze(feat, [0, 2])  # [1, 2, 3, 1, 10]
            # axes is a Variable
            axes = paddle.assign([0, 2])
            out = paddle.squeeze(feat, axes)
            out2 = paddle.fluid.layers.squeeze(feat, axes)

            sgd = paddle.optimizer.SGD()
            sgd.minimize(paddle.mean(out))
            self.assertTrue("Var[" in str(main_prog))

            exe = paddle.static.Executor()
            exe.run(starup_prog)
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
        main_prog = Program()
        starup_prog = Program()
        with program_guard(main_prog, starup_prog):
            fc = paddle.nn.Linear(4, 10)
            x = paddle.randn([2, 3, 4])
            x.stop_gradient = False
            feat = fc(x)  # [2,3,10]
            feat = paddle.unsqueeze(feat, [0, 2])  # [1, 2, 3, 1, 10]
            # axes is a list[Variable]
            axes = [
                paddle.full([1], 0, dtype='int32'),
                paddle.full([1], 2, dtype='int32')
            ]
            out = paddle.squeeze(feat, axes)
            out2 = paddle.fluid.layers.squeeze(feat, axes)

            sgd = paddle.optimizer.SGD()
            sgd.minimize(paddle.mean(out))
            self.assertTrue("Vars[" in str(main_prog))

            exe = paddle.static.Executor()
            exe.run(starup_prog)
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
