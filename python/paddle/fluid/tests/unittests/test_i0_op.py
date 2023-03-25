#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.core as core

paddle.enable_static()
np.random.seed(100)
paddle.seed(100)


def output_numpy_i0(x):
    return np.i0(x)


class TestI0API(unittest.TestCase):
    def setUp(self):
        self.x = np.array([0, 1, 2, 3, 4, 5]).astype("float64")
        self.place = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def test_api_static(self):
        def run(place):
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data(
                    name="x", shape=self.x.shape, dtype='float64'
                )
                y = paddle.i0(x)
                exe = paddle.static.Executor(place)
                res = exe.run(
                    paddle.static.default_main_program(),
                    feed={"x": self.x},
                    fetch_list=[y],
                )
                out_ref = output_numpy_i0(x)
                np.testing.assert_allclose(out_ref, res[0], rtol=1e-5)

        for place in self.place:
            run(place)

    def test_api_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            x = paddle.to_tensor(self.x)
            out = paddle.i0(x)

            out_ref = output_numpy_i0(x)
            np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-5)
            paddle.enable_static()

        for place in self.place:
            run(place)

    def test_empty_input_error(self):
        for place in self.place:
            paddle.disable_static(place)
            x = paddle.to_tensor(self.x, dtype=paddle.int64)
            self.assertRaises(AttributeError, paddle.i0, x)
            paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
