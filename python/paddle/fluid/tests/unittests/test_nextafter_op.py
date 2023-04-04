# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid import core

np.random.seed(10)


def numpy_nextafter(x, y):
    return np.nextafter(x, y)


class TestNextafterAPI(unittest.TestCase):
    def setUp(self):
        self.x = np.random.rand(2, 1, 3).astype("float32")
        self.y = np.random.rand(2, 1, 3).astype("float32")
        self.place = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def test_api_static(self):
        paddle.enable_static()

        def run(place):
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data(
                    'x',
                    shape=self.x.shape,
                    dtype="float32",
                )
                y = paddle.static.data('y', shape=self.y.shape, dtype="float32")
                out1 = paddle.nextafter(x, y)
                exe = paddle.static.Executor(place)
                res = exe.run(
                    feed={'x': self.x, 'y': self.y},
                    fetch_list=[out1],
                )
            out_ref = numpy_nextafter(self.x, self.y)
            np.testing.assert_allclose(out_ref, res[0], rtol=1e-05)

        for place in self.place:
            run(place)

    def test_api_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            x = paddle.to_tensor(self.x)
            y = paddle.to_tensor(self.y)
            out1 = paddle.nexafter(x, y)

            out_ref1 = numpy_nextafter(self.x, self.y)
            np.testing.assert_allclose(out_ref1, out1.numpy(), rtol=1e-05)
            paddle.enable_static()

        for place in self.place:
            run(place)

    def test_empty_input_error(self):
        for place in self.place:
            paddle.disable_static(place)
            x = paddle.to_tensor(self.x)
            y = paddle.to_tensor(self.y)
            self.assertRaises(AttributeError, paddle.nextafter, None, x)
            self.assertRaises(AttributeError, paddle.nextafter, x, None)


if __name__ == "__main__":
    unittest.main()
