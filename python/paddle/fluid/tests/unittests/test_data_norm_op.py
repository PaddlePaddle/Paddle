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
"""This is unit test of Test data_norm Op."""

from __future__ import print_function

import unittest
import numpy as np
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.fluid as fluid
from op_test import OpTest
from paddle.fluid.framework import grad_var_name


def _reference_testing(x, batch_size, batch_sum, batch_square_sum):
    x_shape = x.shape
    means_arr = batch_sum / batch_size
    scales_arr = np.sqrt(batch_size / batch_square_sum)
    for i in range(x_shape[0]):
        x[i] -= means_arr
        x[i] *= scales_arr
    y = np.array(x)
    return y


def create_or_get_tensor(scope, var_name, var, place):
    tensor = scope.var(var_name).get_tensor()
    if var is not None:
        assert isinstance(var, np.ndarray)
        tensor.set_recursive_sequence_lengths([])
        tensor.set(var, place)
    return tensor


class TestDataNormOpInference(unittest.TestCase):
    """
    test class for data norm op
    test forward
    """

    def setUp(self):
        """
        init members of this class
        """
        self.dtype = np.float32
        self.use_mkldnn = False

    def __assert_close(self, tensor, np_array, msg, atol=1e-4):
        self.assertTrue(np.allclose(np.array(tensor), np_array, atol=atol), msg)

    def check_with_place(self, place, data_layout, dtype, shape):
        """
        do forward and check

        Args:
            place(Place): CPUPlace
            data_layout(str): NCHW or NWHC
            dtype(dtype): np.float32
            shape(list): input shape

        """
        epsilon = 0.00001
        if len(shape) == 2:
            x_shape = shape
            c = x_shape[1]
        else:
            ValueError("len(shape) should be equal to 2")
        scale_shape = [c]

        x_val = np.random.random_sample(x_shape).astype(dtype)
        x_val = x_val - 0.5
        batch_size = np.ones(scale_shape).astype(np.float32)
        batch_size *= 1e4
        batch_sum = np.zeros(scale_shape).astype(np.float32)
        batch_square_sum = np.ones(scale_shape).astype(np.float32)
        batch_square_sum *= 1e4

        y_out = _reference_testing(x_val, batch_size, batch_sum,
                                   batch_square_sum).astype(dtype)

        scope = core.Scope()

        # create input
        x_tensor = create_or_get_tensor(scope, "x_val",
                                        OpTest.np_dtype_to_fluid_dtype(x_val),
                                        place)
        batch_size_tensor = create_or_get_tensor(
            scope, "batch_size",
            OpTest.np_dtype_to_fluid_dtype(batch_size), place)
        batch_sum_tensor = create_or_get_tensor(
            scope, "batch_sum",
            OpTest.np_dtype_to_fluid_dtype(batch_sum), place)
        batch_square_sum_tensor = create_or_get_tensor(
            scope, "batch_square_sum",
            OpTest.np_dtype_to_fluid_dtype(batch_square_sum), place)

        # create output
        y_tensor = create_or_get_tensor(scope, "y_out", None, place)
        mean_tensor = create_or_get_tensor(scope, "mean", None, place)
        scales_tensor = create_or_get_tensor(scope, "scales", None, place)

        data_norm_op = Operator(
            "data_norm",
            # inputs
            X="x_val",
            BatchSize="batch_size",
            BatchSum="batch_sum",
            BatchSquareSum="batch_square_sum",
            # outputs
            Y="y_out",
            Means="mean",
            Scales="scales",
            # attrs
            epsilon=epsilon,
            use_mkldnn=self.use_mkldnn)

        data_norm_op.run(scope, place)

        # check inference result
        self.__assert_close(
            y_tensor,
            y_out,
            "inference output are different at " + str(place) + ", " +
            data_layout + ", " + str(np.dtype(dtype)) +
            str(np.array(y_tensor)) + str(y_out),
            atol=1e-3)

    def test_check_output(self):
        """
        test check forward, check output
        """
        places = [core.CPUPlace()]
        for place in places:
            for data_format in ["NCHW", "NHWC"]:
                self.check_with_place(place, data_format, self.dtype, [2, 3])


class TestDataNormOp(OpTest):
    """
    test class for data norm op
    test forward and backward
    """

    def setUp(self):
        """
        init data norm op test env
        """
        self.op_type = 'data_norm'
        self.use_mkldnn = False
        epsilon = 0.00001
        x_shape = [2, 3]
        scale_shape = [3]
        tp = np.float32

        x_val = np.array([[-0.35702616, -0.42756206, -0.08306625],
                          [0.41199666, -0.21719968, -0.10180971]]).astype(tp)
        batch_size = np.ones(scale_shape).astype(tp)
        batch_size *= 1e4
        batch_sum = np.zeros(scale_shape).astype(tp)
        batch_square_sum = np.ones(scale_shape).astype(tp)
        batch_square_sum *= 1e4

        y = np.array(x_val)

        mean = np.array([[0, 0, 0], [0, 0, 0]]).astype(tp)
        scale = np.array([[1, 1, 1], [1, 1, 1]]).astype(tp)

        self.inputs = {
            "X": x_val,
            "BatchSize": batch_size,
            "BatchSum": batch_sum,
            "BatchSquareSum": batch_square_sum
        }
        self.outputs = {"Y": y, "Means": mean, "Scales": scale}
        self.attrs = {"epsilon": epsilon, "use_mkldnn": self.use_mkldnn}

    def test_check_output(self):
        """
        test check forward, check output
        """
        self.check_output()

    def test_check_grad(self):
        """
        test check backward, check grad
        """
        self.check_grad(['X'], 'Y', no_grad_set=set([]))


if __name__ == '__main__':
    unittest.main()
