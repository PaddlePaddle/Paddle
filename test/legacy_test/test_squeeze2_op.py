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

import numpy as np
from op_test import OpTest, convert_float_to_uint16
from test_attribute_var import UnittestBase

import paddle
from paddle.base import core
from paddle.base.framework import Program, program_guard

paddle.enable_static()


# Correct: General.
class TestSqueezeOp(OpTest):
    def setUp(self):
        self.op_type = "squeeze2"
        self.prim_op_type = "comp"
        self.python_api = paddle.squeeze
        self.public_python_api = paddle.squeeze
        self.python_out_sig = [
            "Out"
        ]  # python out sig is customized output signature.
        self.init_test_case()
        self.init_dtype()
        self.if_enable_cinn()
        x = np.random.random(self.ori_shape).astype("float64")
        xshape = np.random.random(self.ori_shape).astype("float64")
        if hasattr(self, "dtype") and self.dtype == np.uint16:
            x = convert_float_to_uint16(x.astype(np.float32))
            xshape = convert_float_to_uint16(xshape.astype(np.float32))
        self.inputs = {"X": x}
        self.init_attrs()
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.new_shape),
            "XShape": xshape,
        }

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        self.check_output(no_check_set=['XShape'], check_prim=True)

    def test_check_grad(self):
        self.check_grad(["X"], "Out", check_prim=True)

    def init_dtype(self):
        self.dtype = np.float64

    def init_test_case(self):
        self.ori_shape = (1, 3, 1, 40)
        self.axes = (0, 2)
        self.new_shape = (3, 40)

    def init_attrs(self):
        self.attrs = {"axes": self.axes}


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and do not support bfloat16",
)
class TestSqueezeOpBF16OP(TestSqueezeOp):
    def init_dtype(self):
        self.dtype = np.uint16


# Correct: There is mins axis.
class TestSqueezeOp1(TestSqueezeOp):
    def init_test_case(self):
        self.ori_shape = (1, 20, 1, 5)
        self.axes = (0, -2)
        self.new_shape = (20, 5)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and do not support bfloat16",
)
class TestSqueezeOp1BF16Op(TestSqueezeOp):
    def init_dtype(self):
        self.dtype = np.uint16


class TestSqueezeOp_ZeroDim1(TestSqueezeOp):
    def init_test_case(self):
        self.ori_shape = ()
        self.axes = (0,)
        self.new_shape = ()


class TestSqueezeOp_ZeroDim2(TestSqueezeOp):
    def init_test_case(self):
        self.ori_shape = (1, 1, 1)
        self.axes = (0, 1, 2)
        self.new_shape = ()


# Correct: No axes input.
class TestSqueezeOp2(TestSqueezeOp):
    def setUp(self):
        self.op_type = "squeeze2"
        self.prim_op_type = "comp"
        self.python_api = paddle.squeeze
        self.public_python_api = paddle.squeeze
        self.python_out_sig = [
            "Out"
        ]  # python out sig is customized output signature.
        self.init_test_case()
        self.init_dtype()
        self.if_enable_cinn()
        x = np.random.random(self.ori_shape).astype("float64")
        xshape = np.random.random(self.ori_shape).astype("float64")
        if hasattr(self, "dtype") and self.dtype == np.uint16:
            x = convert_float_to_uint16(x.astype(np.float32))
            xshape = convert_float_to_uint16(xshape.astype(np.float32))
        self.inputs = {"X": x}
        self.init_attrs()
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.new_shape),
            "XShape": xshape,
        }

    def if_enable_cinn(self):
        pass

    def init_dtype(self):
        self.dtype = np.float64

    def init_test_case(self):
        self.ori_shape = (1, 20, 1, 5)
        self.axes = ()
        self.new_shape = (20, 5)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and do not support bfloat16",
)
class TestSqueezeOp2BF16Op(TestSqueezeOp):
    def init_dtype(self):
        self.dtype = np.uint16


# Correct: Just part of axes be squeezed.
class TestSqueezeOp3(TestSqueezeOp):
    def init_test_case(self):
        self.ori_shape = (6, 1, 5, 1, 4, 1)
        self.axes = (1, -1)
        self.new_shape = (6, 5, 1, 4)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and do not support bfloat16",
)
class TestSqueezeOp3BF16Op(TestSqueezeOp):
    def init_dtype(self):
        self.dtype = np.uint16


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
            out2 = paddle.squeeze(feat, axes)

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
                paddle.full([1], 2, dtype='int32'),
            ]
            out = paddle.squeeze(feat, axes)
            out2 = paddle.squeeze(feat, axes)

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


# test api
class TestSqueezeAPI(unittest.TestCase):
    def setUp(self):
        self.executed_api()

    def executed_api(self):
        self.squeeze = paddle.squeeze

    def test_api(self):
        paddle.disable_static()
        input_data = np.random.random([3, 2, 1]).astype("float32")
        x = paddle.to_tensor(input_data)
        out = self.squeeze(x, axis=2)
        out.backward()

        self.assertEqual(out.shape, [3, 2])

        paddle.enable_static()

    def test_error(self):
        def test_axes_type():
            x2 = paddle.static.data(name="x2", shape=[2, 1, 25], dtype="int32")
            self.squeeze(x2, axis=2.1)

        self.assertRaises(TypeError, test_axes_type)


class TestSqueezeInplaceAPI(TestSqueezeAPI):
    def executed_api(self):
        self.squeeze = paddle.squeeze_


if __name__ == "__main__":
    unittest.main()
