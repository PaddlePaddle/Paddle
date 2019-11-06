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
from op_test import OpTest

import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid.op import Operator


class TestSamplingIdOp(OpTest):
    def setUp(self):
        self.op_type = "sampling_id"
        self.use_mkldnn = False
        self.init_kernel_type()
        self.X = np.random.random((100, 10)).astype('float32')
        self.inputs = {"X": self.X}
        self.Y = np.random.random(100).astype('int64')
        self.outputs = {'Out': self.Y}
        self.attrs = {'max': 1.0, 'min': 0.0, 'seed': 1}

    def test_check_output(self):
        self.check_output_customized(self.verify_output)
        y1 = self.out
        self.check_output_customized(self.verify_output)
        y2 = self.out

        # check dtype
        assert y1.dtype == np.int64
        assert y2.dtype == np.int64

        # check output is index ids of inputs
        inputs_ids = np.arange(self.X.shape[1])
        assert np.isin(y1, inputs_ids).all()
        assert np.isin(y2, inputs_ids).all()

        self.assertTrue(np.array_equal(y1, y2))
        self.assertEqual(len(y1), len(self.Y))

    def verify_output(self, outs):
        out = np.array(outs[0])
        self.out = out

    def init_kernel_type(self):
        pass


class TestSamplingIdShape(unittest.TestCase):
    def test_shape(self):
        x = fluid.layers.data(name='x', shape=[3], dtype='float32')
        output = fluid.layers.sampling_id(x)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place=place)
        exe.run(fluid.default_startup_program())

        feed = {
            'x': np.array(
                [[0.2, 0.3, 0.5], [0.2, 0.3, 0.4]], dtype='float32')
        }
        output_np = exe.run(feed=feed, fetch_list=[output])[0]

        self.assertEqual(output.shape[0], -1)
        self.assertEqual(len(output.shape), 1)
        self.assertEqual(output_np.shape[0], 2)
        self.assertEqual(len(output_np.shape), 1)


if __name__ == "__main__":
    unittest.main()
