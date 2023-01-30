#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid as fluid
import sys

sys.path.append('..')
from op_test import OpTest

paddle.enable_static()


class TestSizeOp(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "size"
        self.shape = []
        self.config()
        input = np.zeros(self.shape, dtype='bool')
        self.inputs = {'Input': input}
        self.outputs = {'Out': np.array([np.size(input)], dtype='int64')}

    def config(self):
        pass

    def test_check_output(self):
        self.check_output_with_place(paddle.device.MLUPlace(0))


class TestRank1Tensor(TestSizeOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def config(self):
        self.shape = [2]


class TestRank2Tensor(TestSizeOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def config(self):
        self.shape = [2, 3]


class TestRank3Tensor(TestSizeOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def config(self):
        self.shape = [2, 3, 100]


class TestLargeTensor(TestSizeOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def config(self):
        self.shape = [2**10]


class TestSizeAPI(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_size_static(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            shape1 = [2, 1, 4, 5]
            shape2 = [1, 4, 5]
            x_1 = paddle.fluid.data(shape=shape1, dtype='int32', name='x_1')
            x_2 = paddle.fluid.data(shape=shape2, dtype='int32', name='x_2')
            input_1 = np.random.random(shape1).astype("int32")
            input_2 = np.random.random(shape2).astype("int32")
<<<<<<< HEAD
            out_1 = paddle.numel(x_1)
            out_2 = paddle.numel(x_2)
            exe = paddle.static.Executor(place=paddle.MLUPlace(0))
            res_1, res_2 = exe.run(
                feed={
                    "x_1": input_1,
                    "x_2": input_2,
                },
                fetch_list=[out_1, out_2],
            )
            assert np.array_equal(
                res_1, np.array([np.size(input_1)]).astype("int64")
            )
            assert np.array_equal(
                res_2, np.array([np.size(input_2)]).astype("int64")
            )
=======
            out_1 = paddle.fluid.layers.size(x_1)
            out_2 = paddle.fluid.layers.size(x_2)
            exe = paddle.static.Executor(place=paddle.MLUPlace(0))
            res_1, res_2 = exe.run(feed={
                "x_1": input_1,
                "x_2": input_2,
            },
                                   fetch_list=[out_1, out_2])
            assert (np.array_equal(res_1,
                                   np.array([np.size(input_1)
                                             ]).astype("int64")))
            assert (np.array_equal(res_2,
                                   np.array([np.size(input_2)
                                             ]).astype("int64")))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_size_imperative(self):
        paddle.disable_static(paddle.MLUPlace(0))
        input_1 = np.random.random([2, 1, 4, 5]).astype("int32")
        input_2 = np.random.random([1, 4, 5]).astype("int32")
        x_1 = paddle.to_tensor(input_1)
        x_2 = paddle.to_tensor(input_2)
<<<<<<< HEAD
        out_1 = paddle.numel(x_1)
        out_2 = paddle.numel(x_2)
        assert np.array_equal(out_1.numpy().item(0), np.size(input_1))
        assert np.array_equal(out_2.numpy().item(0), np.size(input_2))
=======
        out_1 = paddle.fluid.layers.size(x_1)
        out_2 = paddle.fluid.layers.size(x_2)
        assert (np.array_equal(out_1.numpy().item(0), np.size(input_1)))
        assert (np.array_equal(out_2.numpy().item(0), np.size(input_2)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        paddle.enable_static()

    def test_error(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):

            def test_x_type():
                shape = [1, 4, 5]
                input_1 = np.random.random(shape).astype("int32")
<<<<<<< HEAD
                out_1 = paddle.numel(input_1)
=======
                out_1 = paddle.fluid.layers.size(input_1)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            self.assertRaises(TypeError, test_x_type)


if __name__ == '__main__':
    unittest.main()
