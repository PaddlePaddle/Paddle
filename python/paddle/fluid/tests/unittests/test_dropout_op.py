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

import unittest
import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid
from op_test import OpTest


class TestDropoutOp(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.inputs = {'X': np.random.random((32, 64)).astype("float32")}
        self.attrs = {'dropout_prob': 0.0, 'fix_seed': True, 'is_test': False}
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': np.ones((32, 64)).astype('float32')
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out', max_relative_error=0.05)


class TestDropoutOp2(TestDropoutOp):
    def setUp(self):
        self.op_type = "dropout"
        self.inputs = {'X': np.random.random((32, 64)).astype("float32")}
        self.attrs = {'dropout_prob': 1.0, 'fix_seed': True, 'is_test': False}
        self.outputs = {
            'Out': np.zeros((32, 64)).astype('float32'),
            'Mask': np.zeros((32, 64)).astype('float32')
        }


class TestDropoutOp3(TestDropoutOp):
    def setUp(self):
        self.op_type = "dropout"
        self.inputs = {'X': np.random.random((32, 64, 2)).astype("float32")}
        self.attrs = {'dropout_prob': 0.0, 'fix_seed': True, 'is_test': False}
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': np.ones((32, 64, 2)).astype('float32')
        }


class TestDropoutOp4(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.inputs = {'X': np.random.random((32, 64)).astype("float32")}
        self.attrs = {'dropout_prob': 0.35, 'fix_seed': True, 'is_test': True}
        self.outputs = {
            'Out': self.inputs['X'] * (1.0 - self.attrs['dropout_prob'])
        }

    def test_check_output(self):
        self.check_output()


class TestDropoutOp5(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.inputs = {'X': np.random.random((32, 64, 3)).astype("float32")}
        self.attrs = {'dropout_prob': 0.75, 'is_test': True}
        self.outputs = {
            'Out': self.inputs['X'] * (1.0 - self.attrs['dropout_prob'])
        }

    def test_check_output(self):
        self.check_output()


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


#Reference: https://github.com/PaddlePaddle/Paddle/issues/8654
class TestDropoutOp6(unittest.TestCase):
    def program(self):
        x = fluid.layers.data(
            name='x',
            shape=[64, 32, 512],
            dtype='float32',
            append_batch_size=False)
        out = fluid.layers.dropout(x, dropout_prob=0.1, is_test=False)
        return out

    def LoopMaxMin(self, place):
        exe = fluid.Executor(place)
        out = self.program()
        data_input = {}
        in_tensor = fluid.LoDTensor()
        in_tensor.set(np.ones([64, 32, 512], dtype="float32"), place)
        data_input['x'] = in_tensor
        for i in range(10):
            ret_out = exe.run(fluid.framework.default_main_program(),
                              feed=data_input,
                              fetch_list=[out])[0]
            out_max = np.max(ret_out)
            out_min = np.min(ret_out)

            self.assertTrue(isclose(out_max, 1.0))
            self.assertTrue(isclose(out_min, 0.0))

    def test_all(self):
        place = fluid.CUDAPlace(0)
        self.LoopMaxMin(place)

        place = fluid.CPUPlace()
        self.LoopMaxMin(place)


class TestFP16DropoutOp(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.init_test_case()

        x = np.random.random(self.input_size).astype("float16")
        out = x * (1.0 - self.prob)
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.attrs = {
            'dropout_prob': self.prob,
            'fix_seed': self.fix_seed,
            'is_test': True
        }
        self.outputs = {'Out': out}

    def init_test_case(self):
        self.input_size = [32, 64]
        self.prob = 0.35
        self.fix_seed = True

    def test_check_output(self):
        if core.is_compiled_with_cuda() and core.op_support_gpu("dropout"):
            self.check_output_with_place(core.CUDAPlace(0), atol=1e-3)


class TestFP16DropoutOp2(TestFP16DropoutOp):
    def init_test_case(self):
        self.input_size = [32, 64, 3]
        self.prob = 0.75
        self.fix_seed = False


if __name__ == '__main__':
    unittest.main()
