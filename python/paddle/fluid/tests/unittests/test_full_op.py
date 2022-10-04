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
from paddle.fluid.op import Operator
import paddle.fluid as fluid
import paddle
from paddle.fluid import compiler, Program, program_guard
from paddle.fluid.framework import _test_eager_guard


# Test python API
class TestFullAPI(unittest.TestCase):

    def test_api(self):
        positive_2_int32 = fluid.layers.fill_constant([1], "int32", 2)

        positive_2_int64 = fluid.layers.fill_constant([1], "int64", 2)
        shape_tensor_int32 = fluid.data(name="shape_tensor_int32",
                                        shape=[2],
                                        dtype="int32")

        shape_tensor_int64 = fluid.data(name="shape_tensor_int64",
                                        shape=[2],
                                        dtype="int64")

        out_1 = paddle.full(shape=[1, 2], dtype="float32", fill_value=1.1)

        out_2 = paddle.full(shape=[1, positive_2_int32],
                            dtype="float32",
                            fill_value=1.1)

        out_3 = paddle.full(shape=[1, positive_2_int64],
                            dtype="float32",
                            fill_value=1.1)

        out_4 = paddle.full(shape=shape_tensor_int32,
                            dtype="float32",
                            fill_value=1.2)

        out_5 = paddle.full(shape=shape_tensor_int64,
                            dtype="float32",
                            fill_value=1.1)

        out_6 = paddle.full(shape=shape_tensor_int64,
                            dtype=np.float32,
                            fill_value=1.1)

        val = fluid.layers.fill_constant(shape=[1], dtype=np.float32, value=1.1)
        out_7 = paddle.full(shape=shape_tensor_int64,
                            dtype=np.float32,
                            fill_value=val)

        exe = fluid.Executor(place=fluid.CPUPlace())
        res_1, res_2, res_3, res_4, res_5, res_6, res_7 = exe.run(
            fluid.default_main_program(),
            feed={
                "shape_tensor_int32": np.array([1, 2]).astype("int32"),
                "shape_tensor_int64": np.array([1, 2]).astype("int64"),
            },
            fetch_list=[out_1, out_2, out_3, out_4, out_5, out_6, out_7])

        assert np.array_equal(res_1, np.full([1, 2], 1.1, dtype="float32"))
        assert np.array_equal(res_2, np.full([1, 2], 1.1, dtype="float32"))
        assert np.array_equal(res_3, np.full([1, 2], 1.1, dtype="float32"))
        assert np.array_equal(res_4, np.full([1, 2], 1.2, dtype="float32"))
        assert np.array_equal(res_5, np.full([1, 2], 1.1, dtype="float32"))
        assert np.array_equal(res_6, np.full([1, 2], 1.1, dtype="float32"))
        assert np.array_equal(res_7, np.full([1, 2], 1.1, dtype="float32"))

    def test_api_eager(self):
        with fluid.dygraph.base.guard():
            with _test_eager_guard():
                positive_2_int32 = fluid.layers.fill_constant([1], "int32", 2)
                positive_2_int64 = fluid.layers.fill_constant([1], "int64", 2)
                positive_4_int64 = fluid.layers.fill_constant([1], "int64", 4,
                                                              True)

                out_1 = paddle.full(shape=[1, 2],
                                    dtype="float32",
                                    fill_value=1.1)

                out_2 = paddle.full(shape=[1, positive_2_int32.item()],
                                    dtype="float32",
                                    fill_value=1.1)

                out_3 = paddle.full(shape=[1, positive_2_int64.item()],
                                    dtype="float32",
                                    fill_value=1.1)

                out_4 = paddle.full(shape=[1, 2],
                                    dtype="float32",
                                    fill_value=1.2)

                out_5 = paddle.full(shape=[1, 2],
                                    dtype="float32",
                                    fill_value=1.1)

                out_6 = paddle.full(shape=[1, 2],
                                    dtype=np.float32,
                                    fill_value=1.1)

                val = fluid.layers.fill_constant(shape=[1],
                                                 dtype=np.float32,
                                                 value=1.1)
                out_7 = paddle.full(shape=[1, 2],
                                    dtype=np.float32,
                                    fill_value=val)

                out_8 = paddle.full(shape=positive_2_int32,
                                    dtype="float32",
                                    fill_value=1.1)

                out_9 = paddle.full(shape=[
                    positive_2_int32, positive_2_int64, positive_4_int64
                ],
                                    dtype="float32",
                                    fill_value=1.1)

                # test for numpy.float64 as fill_value
                out_10 = paddle.full_like(out_7,
                                          dtype=np.float32,
                                          fill_value=np.abs(1.1))

                assert np.array_equal(out_1,
                                      np.full([1, 2], 1.1, dtype="float32"))
                assert np.array_equal(out_2,
                                      np.full([1, 2], 1.1, dtype="float32"))
                assert np.array_equal(out_3,
                                      np.full([1, 2], 1.1, dtype="float32"))
                assert np.array_equal(out_4,
                                      np.full([1, 2], 1.2, dtype="float32"))
                assert np.array_equal(out_5,
                                      np.full([1, 2], 1.1, dtype="float32"))
                assert np.array_equal(out_6,
                                      np.full([1, 2], 1.1, dtype="float32"))
                assert np.array_equal(out_7,
                                      np.full([1, 2], 1.1, dtype="float32"))
                assert np.array_equal(out_8, np.full([2], 1.1, dtype="float32"))
                assert np.array_equal(out_9,
                                      np.full([2, 2, 4], 1.1, dtype="float32"))
                assert np.array_equal(out_10,
                                      np.full([1, 2], 1.1, dtype="float32"))


class TestFullOpError(unittest.TestCase):

    def test_errors(self):
        with program_guard(Program(), Program()):
            #for ci coverage
            self.assertRaises(TypeError,
                              paddle.full,
                              shape=[1],
                              fill_value=5,
                              dtype='uint4')

            # The argument dtype of full must be one of bool, float16,
            #float32, float64, uint8, int16, int32 or int64

            # The argument shape's type of full_op  must be list, tuple or Variable.
            def test_shape_type():
                paddle.full(shape=1, dtype="float32", fill_value=1)

            self.assertRaises(TypeError, test_shape_type)

            # The argument shape's size of full_op must not be 0.
            def test_shape_size():
                paddle.full(shape=[], dtype="float32", fill_value=1)

            self.assertRaises(AssertionError, test_shape_size)

            # The shape dtype of full op must be int32 or int64.
            def test_shape_tensor_dtype():
                shape = fluid.data(name="shape_tensor",
                                   shape=[2],
                                   dtype="float32")
                paddle.full(shape=shape, dtype="float32", fill_value=1)

            self.assertRaises(TypeError, test_shape_tensor_dtype)

            def test_shape_tensor_list_dtype():
                shape = fluid.data(name="shape_tensor_list",
                                   shape=[1],
                                   dtype="bool")
                paddle.full(shape=[shape, 2], dtype="float32", fill_value=1)

            self.assertRaises(TypeError, test_shape_tensor_list_dtype)


if __name__ == "__main__":
    unittest.main()
