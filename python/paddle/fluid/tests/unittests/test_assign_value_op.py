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

import paddle.fluid as fluid
import paddle.fluid.layers as layers
import op_test
import numpy
import unittest
import paddle.fluid.framework as framework


class TestAssignValueOp(op_test.OpTest):
    def setUp(self):
        self.op_type = "assign_value"
        x = numpy.random.random(size=(2, 5)).astype(numpy.float32)
        self.inputs = {}
        self.outputs = {'Out': x}
        self.attrs = {
            'shape': x.shape,
            'dtype': framework.convert_np_dtype_to_dtype_(x.dtype),
            'fp32_values': [float(v) for v in x.flat]
        }

    def test_forward(self):
        self.check_output()

    def test_assign(self):
        val = (
            -100 + 200 * numpy.random.random(size=(2, 5))).astype(numpy.int32)
        x = layers.create_tensor(dtype="float32")
        layers.assign(input=val, output=x)
        exe = fluid.Executor(fluid.CPUPlace())
        fetched_x = exe.run(fluid.default_main_program(),
                            feed={},
                            fetch_list=[x])[0]
        self.assertTrue(
            numpy.array_equal(fetched_x, val),
            "fetch_x=%s val=%s" % (fetched_x, val))
        self.assertEqual(fetched_x.dtype, val.dtype)


if __name__ == '__main__':
    unittest.main()
