# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

<<<<<<< HEAD
import os
import unittest

import numpy as np
from op_test import OpTest
from test_attribute_var import UnittestBase

import paddle
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard


class BaseTestCase(OpTest):
=======
from __future__ import print_function

import os
import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard
from test_attribute_var import UnittestBase


class BaseTestCase(OpTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (3, 4, 5)
        self.dtype = 'float32'
        self.axis = 0

    def setUp(self):
        self.initTestCase()
        self.x = (1000 * np.random.random(self.dims)).astype(self.dtype)
        self.inputs = {'X': self.x}
        self.attrs = {'axis': self.axis}
        if self.op_type == "arg_min":
            self.outputs = {'Out': np.argmin(self.x, axis=self.axis)}
        else:
            self.outputs = {'Out': np.argmax(self.x, axis=self.axis)}

    def test_check_output(self):
        self.check_output()


class TestCase0(BaseTestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, 4, 5)
        self.dtype = 'float32'
        self.axis = 0


class TestCase1(BaseTestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (3, 4)
        self.dtype = 'float64'
        self.axis = 1


class TestCase2(BaseTestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, 4)
        self.dtype = 'int64'
        self.axis = 0


<<<<<<< HEAD
@unittest.skipIf(
    not paddle.is_compiled_with_cuda(), "FP16 test runs only on GPU"
)
class TestCase0FP16(BaseTestCase):
=======
@unittest.skipIf(not paddle.is_compiled_with_cuda(),
                 "FP16 test runs only on GPU")
class TestCase0FP16(BaseTestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, 4, 5)
        self.dtype = np.float16
        self.axis = 0


<<<<<<< HEAD
@unittest.skipIf(
    not paddle.is_compiled_with_cuda(), "FP16 test runs only on GPU"
)
class TestCase1FP16(BaseTestCase):
=======
@unittest.skipIf(not paddle.is_compiled_with_cuda(),
                 "FP16 test runs only on GPU")
class TestCase1FP16(BaseTestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (3, 4)
        self.dtype = np.float16
        self.axis = 1


class TestCase2_1(BaseTestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, 4)
        self.dtype = 'int64'
        self.axis = -1


class TestCase3(BaseTestCase):
<<<<<<< HEAD
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3,)
=======

    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.dtype = 'int64'
        self.axis = 0


class TestCase4(BaseTestCase):
<<<<<<< HEAD
    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (1,)
=======

    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (1, )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.dtype = 'int32'
        self.axis = 0


class TestCase3_(BaseTestCase):
<<<<<<< HEAD
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3,)
=======

    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.axis = 0


class BaseTestComplex1_1(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (4, 5, 6)
        self.dtype = 'int32'
        self.axis = 2

    def setUp(self):
        self.initTestCase()
        self.x = (np.random.random(self.dims)).astype(self.dtype)
        self.inputs = {'X': self.x}
        self.attrs = {'axis': self.axis}
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT32)}
        if self.op_type == "arg_min":
            self.outputs = {
                'Out': np.argmin(self.x, axis=self.axis).asdtype("int32")
            }
        else:
            self.outputs = {
                'Out': np.argmax(self.x, axis=self.axis).asdtype("int32")
            }


class BaseTestComplex1_2(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (4, 5, 6)
        self.dtype = 'int32'
        self.axis = 2

    def setUp(self):
        self.initTestCase()
        self.x = (np.random.random(self.dims)).astype(self.dtype)
        self.inputs = {'X': self.x}
        self.attrs = {'axis': self.axis}
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT32)}
        if self.op_type == "arg_min":
            self.outputs = {
                'Out': np.argmin(self.x, axis=self.axis).asdtype("int32")
            }
        else:
            self.outputs = {
                'Out': np.argmax(self.x, axis=self.axis).asdtype("int32")
            }


class BaseTestComplex2_1(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (4, 5, 6)
        self.dtype = 'int32'
        self.axis = 2

    def setUp(self):
        self.initTestCase()
        self.x = (np.random.random(self.dims)).astype(self.dtype)
        self.inputs = {'X': self.x}
        self.attrs = {'axis': self.axis}
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT32)}
        self.attrs = {'keep_dims': True}
        if self.op_type == "arg_min":
            self.outputs = {
<<<<<<< HEAD
                'Out': np.argmin(self.x, axis=self.axis)
                .asdtype("int32")
                .reshape(4, 5, 1)
            }
        else:
            self.outputs = {
                'Out': np.argmax(self.x, axis=self.axis)
                .asdtype("int32")
                .reshape(4, 5, 1)
=======
                'Out':
                np.argmin(self.x,
                          axis=self.axis).asdtype("int32").reshape(4, 5, 1)
            }
        else:
            self.outputs = {
                'Out':
                np.argmax(self.x,
                          axis=self.axis).asdtype("int32").reshape(4, 5, 1)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }


class BaseTestComplex2_2(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (4, 5, 6)
        self.dtype = 'int32'
        self.axis = 2

    def setUp(self):
        self.initTestCase()
        self.x = (np.random.random(self.dims)).astype(self.dtype)
        self.inputs = {'X': self.x}
        self.attrs = {'axis': self.axis}
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT32)}
        self.attrs = {'keep_dims': True}
        if self.op_type == "arg_min":
            self.outputs = {
<<<<<<< HEAD
                'Out': np.argmin(self.x, axis=self.axis)
                .asdtype("int32")
                .reshape(4, 5, 1)
            }
        else:
            self.outputs = {
                'Out': np.argmax(self.x, axis=self.axis)
                .asdtype("int32")
                .reshape(4, 5, 1)
=======
                'Out':
                np.argmin(self.x,
                          axis=self.axis).asdtype("int32").reshape(4, 5, 1)
            }
        else:
            self.outputs = {
                'Out':
                np.argmax(self.x,
                          axis=self.axis).asdtype("int32").reshape(4, 5, 1)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }


class TestArgMaxTensorAxis(UnittestBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_info(self):
        self.shapes = [[2, 3, 4]]
        self.x = [np.random.randn(*shape) for shape in self.shapes]
        self.save_path = os.path.join(self.temp_dir.name, self.path_prefix())

    def test_static(self):
        main_prog = Program()
        starup_prog = Program()
        with program_guard(main_prog, starup_prog):
            fc = paddle.nn.Linear(4, 10)
            x = paddle.randn([2, 3, 4])
            x.stop_gradient = False
            feat = fc(x)

            out = self.call_func(feat)

            sgd = paddle.optimizer.SGD()
            sgd.minimize(paddle.mean(paddle.cast(out, 'float32')))
            self.assertTrue(self.var_prefix() in str(main_prog))

            exe = paddle.static.Executor()
            exe.run(starup_prog)
            res = exe.run(fetch_list=[feat, out])
<<<<<<< HEAD
            paddle.static.save_inference_model(
                self.save_path, [x], [feat, out], exe
            )
=======
            paddle.static.save_inference_model(self.save_path, [x], [feat, out],
                                               exe)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            gt = np.argmax(res[0], 0)
            np.testing.assert_allclose(res[1], gt)

            # Test for Inference Predictor
            infer_outs = self.infer_prog()
            gt = np.argmax(infer_outs[0], 0)
            np.testing.assert_allclose(infer_outs[1], gt)

    def path_prefix(self):
        return 'argmax_tensor_axis'

    def var_prefix(self):
        return "Var["

    def call_func(self, x):
        axis = paddle.assign(0)
        out = paddle.argmax(x, axis)
        return out


class TestArgMinTensorAxis(TestArgMaxTensorAxis):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_static(self):
        main_prog = Program()
        starup_prog = Program()
        with program_guard(main_prog, starup_prog):
            fc = paddle.nn.Linear(4, 10)
            x = paddle.randn([2, 3, 4])
            x.stop_gradient = False
            feat = fc(x)
            feat = paddle.cast(feat, 'int32')
            out = self.call_func(feat)

            sgd = paddle.optimizer.SGD()
            sgd.minimize(paddle.mean(paddle.cast(out, 'float32')))
            self.assertTrue(self.var_prefix() in str(main_prog))

            exe = paddle.static.Executor()
            exe.run(starup_prog)
            res = exe.run(fetch_list=[feat, out])
<<<<<<< HEAD
            paddle.static.save_inference_model(
                self.save_path, [x], [feat, out], exe
            )
=======
            paddle.static.save_inference_model(self.save_path, [x], [feat, out],
                                               exe)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            gt = np.argmin(res[0], 1)
            np.testing.assert_allclose(np.squeeze(res[1]), gt)

            # Test for Inference Predictor
            infer_outs = self.infer_prog()
            gt = np.argmin(infer_outs[0], 1)
            np.testing.assert_allclose(np.squeeze(infer_outs[1]), gt)

    def path_prefix(self):
        return 'argmin_tensor_axis'

    def call_func(self, x):
        axis = paddle.assign(1)
        out = paddle.argmin(x, axis, keepdim=True)
        return out


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
