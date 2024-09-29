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
from paddle import base
from paddle.base import core
from paddle.base.executor import Executor


class TestSquareErrorCost(unittest.TestCase):

    def test_square_error_cost(self):
        paddle.enable_static()
        shape = [2, 3]
        input_val = np.random.uniform(0.1, 0.5, shape).astype("float32")
        label_val = np.random.uniform(0.1, 0.5, shape).astype("float32")

        sub = input_val - label_val
        np_result = sub * sub

        for use_cuda in (
            [False, True] if core.is_compiled_with_cuda() else [False]
        ):
            with paddle.static.program_guard(paddle.static.Program()):
                input_var = paddle.static.data(
                    name="input", shape=shape, dtype="float32"
                )
                label_var = paddle.static.data(
                    name="label", shape=shape, dtype="float32"
                )
                output = paddle.nn.functional.square_error_cost(
                    input=input_var, label=label_var
                )

                place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
                exe = Executor(place)
                (result,) = exe.run(
                    paddle.static.default_main_program(),
                    feed={"input": input_val, "label": label_val},
                    fetch_list=[output],
                )

                np.testing.assert_allclose(np_result, result, rtol=1e-05)


class TestSquareErrorInvalidInput(unittest.TestCase):

    def test_error(self):
        def test_invalid_input():
            input = [256, 3]
            label = paddle.static.data(
                name='label1', shape=[None, 3], dtype='float32'
            )
            loss = paddle.nn.functional.square_error_cost(input, label)

        self.assertRaises(TypeError, test_invalid_input)

        def test_invalid_label():
            input = paddle.static.data(
                name='input2', shape=[None, 3], dtype='float32'
            )
            label = [256, 3]
            loss = paddle.nn.functional.square_error_cost(input, label)

        self.assertRaises(TypeError, test_invalid_label)


if __name__ == "__main__":
    unittest.main()
