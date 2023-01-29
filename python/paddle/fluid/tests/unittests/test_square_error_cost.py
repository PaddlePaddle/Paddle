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

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.executor import Executor


class TestSquareErrorCost(unittest.TestCase):
    def test_square_error_cost(self):
        input_val = np.random.uniform(0.1, 0.5, (2, 3)).astype("float32")
        label_val = np.random.uniform(0.1, 0.5, (2, 3)).astype("float32")

        sub = input_val - label_val
        np_result = sub * sub

        input_var = paddle.tensor.create_tensor(dtype="float32", name="input")
        label_var = paddle.tensor.create_tensor(dtype="float32", name="label")
        output = paddle.nn.functional.square_error_cost(
            input=input_var, label=label_var
        )

        for use_cuda in (
            [False, True] if core.is_compiled_with_cuda() else [False]
        ):

            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = Executor(place)
            (result,) = exe.run(
                fluid.default_main_program(),
                feed={"input": input_val, "label": label_val},
                fetch_list=[output],
            )

            np.testing.assert_allclose(np_result, result, rtol=1e-05)


class TestSquareErrorInvalidInput(unittest.TestCase):
    def test_error(self):
        def test_invalid_input():
            input = [256, 3]
            label = fluid.data(name='label1', shape=[None, 3], dtype='float32')
            loss = paddle.nn.functional.square_error_cost(input, label)

        self.assertRaises(TypeError, test_invalid_input)

        def test_invalid_label():
            input = fluid.data(name='input2', shape=[None, 3], dtype='float32')
            label = [256, 3]
            loss = paddle.nn.functional.square_error_cost(input, label)

        self.assertRaises(TypeError, test_invalid_label)


if __name__ == "__main__":
    unittest.main()
