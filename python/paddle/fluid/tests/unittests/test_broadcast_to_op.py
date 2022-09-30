#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
import paddle

paddle.enable_static()


class TestBroadcastToError(unittest.TestCase):

    def test_errors(self):
        with program_guard(Program(), Program()):
            x1 = fluid.create_lod_tensor(np.array([[-1]]), [[1]],
                                         fluid.CPUPlace())
            shape = [2, 2]
            self.assertRaises(TypeError, paddle.tensor.broadcast_to, x1, shape)
            x2 = fluid.layers.data(name='x2', shape=[4], dtype="uint8")
            self.assertRaises(TypeError, paddle.tensor.broadcast_to, x2, shape)
            x3 = fluid.layers.data(name='x3', shape=[4], dtype="bool")
            x3.stop_gradient = False
            self.assertRaises(ValueError, paddle.tensor.broadcast_to, x3, shape)


# Test python API
class TestBroadcastToAPI(unittest.TestCase):

    def test_api(self):
        input = np.random.random([12, 14]).astype("float32")
        x = fluid.layers.data(name='x',
                              shape=[12, 14],
                              append_batch_size=False,
                              dtype="float32")

        positive_2 = fluid.layers.fill_constant([1], "int32", 12)
        expand_shape = fluid.layers.data(name="expand_shape",
                                         shape=[2],
                                         append_batch_size=False,
                                         dtype="int32")

        out_1 = paddle.broadcast_to(x, shape=[12, 14])
        out_2 = paddle.broadcast_to(x, shape=[positive_2, 14])
        out_3 = paddle.broadcast_to(x, shape=expand_shape)

        g0 = fluid.backward.calc_gradient(out_2, x)

        exe = fluid.Executor(place=fluid.CPUPlace())
        res_1, res_2, res_3 = exe.run(fluid.default_main_program(),
                                      feed={
                                          "x":
                                          input,
                                          "expand_shape":
                                          np.array([12, 14]).astype("int32")
                                      },
                                      fetch_list=[out_1, out_2, out_3])
        assert np.array_equal(res_1, np.tile(input, (1, 1)))
        assert np.array_equal(res_2, np.tile(input, (1, 1)))
        assert np.array_equal(res_3, np.tile(input, (1, 1)))


if __name__ == "__main__":
    unittest.main()
