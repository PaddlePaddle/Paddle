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
from op_test import (
    paddle_static_guard,
)

import paddle
from paddle.base import Program, program_guard


class TestEmbedOpError(unittest.TestCase):
    def test_errors(self):
        with paddle_static_guard():
            with program_guard(Program(), Program()):
                input_data = np.random.randint(0, 10, (4, 1)).astype("int64")

                def test_Variable():
                    # the input type must be Variable
                    paddle.static.nn.embedding(input=input_data, size=(10, 64))

                self.assertRaises(TypeError, test_Variable)

                def test_input_dtype():
                    # the input dtype must be int64
                    input = paddle.static.data(
                        name='x', shape=[4, 1], dtype='float32'
                    )
                    paddle.static.nn.embedding(input=input, size=(10, 64))

                self.assertRaises(TypeError, test_input_dtype)

                def test_param_dtype():
                    # dtype must be float32 or float64
                    input2 = paddle.static.data(
                        name='x2', shape=[4, 1], dtype='int64'
                    )
                    paddle.static.nn.embedding(
                        input=input2, size=(10, 64), dtype='int64'
                    )

                self.assertRaises(TypeError, test_param_dtype)

                input3 = paddle.static.data(
                    name='x3', shape=[4, 1], dtype='int64'
                )
                paddle.static.nn.embedding(
                    input=input3, size=(10, 64), dtype='float16'
                )


if __name__ == "__main__":
    unittest.main()
