# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest
import numpy as np

import paddle
import paddle.fluid.core as core
from paddle.vision.ops import random_flip


class TestRandomFlip(unittest.TestCase):
    def test_errors(self):
        try:
            data = paddle.ones([16, 3, 32, 32], dtype="float32")
            out = random_flip(data, 1.5)

            # should not execute following lines
            assert False
        except ValueError:
            pass

    def test_output_dynamic(self):
        data = paddle.ones([16, 3, 32, 32], dtype="float32")
        out = random_flip(data, 0.5)

        assert out.dtype == paddle.bool
        assert out.shape == [16, 1]

    def test_output_static(self):
        paddle.enable_static()
        input_data = paddle.static.data(shape=[16, 3, 32, 32], dtype="float32", name="input")
        out_data = random_flip(input_data, 0.5)

        places = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))

        for place in places:
            exe = paddle.static.Executor(place)
            out, = exe.run(paddle.static.default_main_program(),
                    feed={"input": np.ones([16, 3, 32, 32], dtype="float32")},
                    fetch_list=[out_data])
            assert out.dtype == np.bool
            assert out.shape == (16, 1)
        paddle.disable_static()


if __name__ == '__main__':
    unittest.main()
