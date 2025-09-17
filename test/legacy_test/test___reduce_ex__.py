# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import pickle
import unittest

import numpy as np

import paddle


class Test__Reduce_EX__BASE(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.dtypes = [
            'bool',
            'float16',
            'bfloat16',
            'uint16',
            'float32',
            'float64',
            'int4',
            'int8',
            'int16',
            'int32',
            'int64',
            'uint8',
        ]
        self.places = [paddle.CPUPlace()]
        if paddle.device.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))
        self.shape = [3, 4, 5, 6]

    def _prepare_data(self, dtype, place):
        if dtype.startswith("int") or dtype.startswith("uint"):
            tensor = paddle.randint(low=0, high=10, shape=self.shape)
        elif (
            dtype.startswith("float")
            or dtype.startswith("bfloat")
            or dtype.startswith("complex")
        ):
            tensor = paddle.rand(shape=self.shape).astype(dtype)
        elif dtype.startswith("bool"):
            tensor = paddle.rand(self.shape) > 0.5

        return paddle.tensor(tensor, device=place)

    def _perform_compare(self, actual, expected):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert actual.place == expected.place
        assert actual.stop_gradient == expected.stop_gradient
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())

    def _perform_test(self, place, dtype, pin_mem, requires_grad):
        x = paddle.tensor(self._prepare_data(dtype, place))
        x.requires_grad = requires_grad
        if pin_mem:
            x = x.pin_memory()
        data = pickle.dumps(x)
        y = pickle.loads(data)
        self._perform_compare(x, y)

    def test___reduce_ex__(self):
        for place in self.places:
            for dtype in self.dtypes:
                for pin_mem in (
                    [True, False]
                    if paddle.device.is_compiled_with_cuda()
                    else [False]
                ):
                    for requires_grad in [True, False]:
                        self._perform_test(place, dtype, pin_mem, requires_grad)


if __name__ == '__main__':
    unittest.main()
