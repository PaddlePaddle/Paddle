#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import paddle
import paddle.fluid as fluid
import unittest
import numpy as np
from op_test import OpTest
from paddle.fluid import Program, program_guard
from paddle.fluid.layer_helper import LayerHelper


class TestLogSumOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):

            x1 = fluid.layers.data(name='x1', shape=[120], dtype="uint8")
            self.assertRaises(Exception, paddle.logsumexp, x1)

            x2 = fluid.layers.data(name='x2', shape=[2, 3], dtype="int")
            self.assertRaises(Exception, paddle.logsumexp, x2)

            x3 = fluid.layers.data(name='x3', shape=[3], dtype="float16")
            self.assertRaises(Exception, paddle.logsumexp, x3)

            self.assertRaises(AssertionError, paddle.logsumexp, None)


class TestLogSumExpOp(unittest.TestCase):
    def test_dygraph(self):
        with fluid.dygraph.guard():
            np_x = np.random.uniform(0.1, 1, [123]).astype(np.float32)
            x = fluid.dygraph.to_variable(np_x)
            self.assertTrue(
                np.allclose(
                    paddle.logsumexp(x).numpy(), np.log(np.sum(np.exp(np_x)))))

            np_x = np.random.uniform(0.1, 1, [2, 3, 4]).astype(np.float32)
            x = fluid.dygraph.to_variable(np_x)
            self.assertTrue(
                np.allclose(
                    paddle.logsumexp(
                        x, dim=[1, 2]).numpy(),
                    np.log(np.sum(np.exp(np_x), axis=(1, 2)))))

            np_x = np.random.uniform(0.1, 1, [2, 3, 4]).astype(np.float32)
            x = fluid.dygraph.to_variable(np_x)
            self.assertTrue(
                np.allclose(
                    paddle.logsumexp(
                        x, dim=[2]).numpy(),
                    np.log(np.sum(np.exp(np_x), axis=(2)))))

            np_x = np.random.uniform(0.1, 1, [2, 3, 4]).astype(np.float32)
            x = fluid.dygraph.to_variable(np_x)
            self.assertTrue(
                np.allclose(
                    paddle.logsumexp(
                        x, keepdim=True).numpy(),
                    np.log(np.sum(np.exp(np_x), keepdims=True))))

            np_x = np.random.uniform(0.1, 1, [2, 3, 4]).astype(np.float32)
            x = fluid.dygraph.to_variable(np_x)
            helper = LayerHelper("test_logsumexp")
            out = helper.create_variable(
                type=x.type, name='out', dtype=x.dtype, persistable=False)
            paddle.logsumexp(x, out=out)
            self.assertTrue(
                np.allclose(out.numpy(), np.log(np.sum(np.exp(np_x)))))


if __name__ == '__main__':
    unittest.main()
