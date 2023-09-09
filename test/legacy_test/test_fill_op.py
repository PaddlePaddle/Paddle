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
from eager_op_test import OpTest, convert_float_to_uint16
from op import Operator

from paddle.base import core


class TestFillOp1(OpTest):
    def setUp(self):
        self.op_type = "fill"
        self.init_dtype()
        val = np.random.random(size=[100, 200])
        self.inputs = {}
        self.attrs = {
            'value': val.flatten().tolist(),
            'shape': [100, 200],
            'dtype': int(core.VarDesc.VarType.FP64),
            'force_cpu': False,
        }
        self.outputs = {'Out': val.astype('float64')}

    def init_dtype(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output()


class TestFillOp2(OpTest):
    def setUp(self):
        self.op_type = "fill"
        val = np.random.random(size=[100, 200])
        self.inputs = {}
        self.attrs = {
            'value': val.flatten().tolist(),
            'shape': [100, 200],
            'dtype': int(core.VarDesc.VarType.FP64),
            'force_cpu': True,
        }
        self.outputs = {'Out': val.astype('float64')}

    def test_check_output(self):
        self.check_output()


class TestFillOp3(unittest.TestCase):
    def check_with_place(self, place, f_cpu):
        scope = core.Scope()
        # create Out Variable
        out = scope.var('Out').get_tensor()

        # create and run fill_op operator
        val = np.random.random(size=[300, 200])
        fill_op = Operator(
            "fill",
            value=val.flatten(),
            shape=[300, 200],
            dtype=int(core.VarDesc.VarType.FP32),
            force_cpu=f_cpu,
            Out='Out',
        )
        fill_op.run(scope, place)

        # get result from Out
        result_array = np.array(out)
        full_array = np.array(val, 'float32')

        np.testing.assert_array_equal(result_array, full_array)

    def test_fill_op(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))

        for place in places:
            self.check_with_place(place, True)
            self.check_with_place(place, False)


class TestFillFP16OP(TestFillOp1):
    def init_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not complied with CUDA and not support the bfloat16",
)
class TestFillBF16OP(OpTest):
    def setUp(self):
        self.op_type = "fill"
        self.dtype = np.uint16
        val = np.random.random(size=[100, 200])
        self.inputs = {}
        self.attrs = {
            'value': val.flatten().tolist(),
            'shape': [100, 200],
            'dtype': int(core.VarDesc.VarType.BF16),
            'force_cpu': False,
        }
        self.outputs = {'Out': convert_float_to_uint16(val)}

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place)


if __name__ == '__main__':
    unittest.main()
