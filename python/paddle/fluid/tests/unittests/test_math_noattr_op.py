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

from __future__ import print_function

import unittest

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

from op_test import OpTest


class TestMathNoattrOp(OpTest):
    def setUp(self):
        self.op_type = "sin"
        self.init_dtype()

        x = np.random.uniform(-1, 1, [10, 12]).astype(self.dtype)
        out = np.sin(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')

    def test_check_output(self):
        self.check_output()

    def init_dtype(self):
        self.dtype = np.float64

    def init_kernel_type(self):
        pass

    def test_out(self):
        with fluid.program_guard(fluid.Program()):
            data = fluid.layers.data(name="X", shape=[1])
            out = eval("paddle.tensor.math.%s(data, out=data)" % self.op_type)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            result = exe.run(feed={"X": np.array([0.1])},
                             fetch_list=[data, out])
            self.assertEqual(result[0], result[1])

    def test_out_name(self):
        with fluid.program_guard(fluid.Program()):
            data = fluid.layers.data(name="X", shape=[1])
            out = eval("paddle.tensor.math.%s(data, name='Y', out=data)" %
                       self.op_type)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            result = exe.run(feed={"X": np.array([0.1])},
                             fetch_list=[data, out])
            self.assertEqual(result[0], result[1])


class TestAtan(TestMathNoattrOp):
    def setUp(self):
        self.op_type = "atan"
        self.init_dtype()

        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.arctan(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestTanh(TestMathNoattrOp):
    def setUp(self):
        self.op_type = "tanh"
        self.init_dtype()
        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.tanh(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')

    def init_dtype(self):
        #TODO If dtype is float64, the output (Out) has diff at CPUPlace
        # when using and not using inplace. Therefore, set dtype as float32
        # for now.
        self.dtype = np.float32


class TestSqrt(TestMathNoattrOp):
    def setUp(self):
        self.op_type = "sqrt"
        self.init_dtype()

        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.sqrt(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


#------------------ Test Cudnn Activation----------------------
def create_test_act_cudnn_class(parent, atol=1e-3, grad_atol=1e-3):
    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    class TestActCudnn(parent):
        def init_kernel_type(self):
            self.attrs = {"use_cudnn": True}

    cls_name = "{0}_{1}".format(parent.__name__, "cudnn")
    TestActCudnn.__name__ = cls_name
    globals()[cls_name] = TestActCudnn


create_test_act_cudnn_class(TestMathNoattrOp)
create_test_act_cudnn_class(TestAtan)
create_test_act_cudnn_class(TestSqrt)
create_test_act_cudnn_class(TestTanh)


#------------------ Test Fp16 ----------------------
def create_test_act_fp16_class(parent,
                               atol=1e-3,
                               grad_check=True,
                               grad_atol=0.80):
    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    class TestActFp16(parent):
        def init_dtype(self):
            self.dtype = np.float16

        def test_check_output(self):
            place = core.CUDAPlace(0)
            support_fp16 = core.is_float16_supported(place)
            if support_fp16:
                self.check_output_with_place(place, atol=atol)

        def test_check_grad(self):
            place = core.CUDAPlace(0)
            support_fp16 = core.is_float16_supported(place)
            if support_fp16 and grad_check:
                self.check_grad_with_place(
                    place, ['X'], 'Out', max_relative_error=grad_atol)

    cls_name = "{0}_{1}".format(parent.__name__, "fp16")
    TestActFp16.__name__ = cls_name
    globals()[cls_name] = TestActFp16


create_test_act_fp16_class(TestMathNoattrOp)
create_test_act_fp16_class(TestAtan)
create_test_act_fp16_class(TestSqrt)
create_test_act_fp16_class(TestTanh)

if __name__ == "__main__":
    unittest.main()
