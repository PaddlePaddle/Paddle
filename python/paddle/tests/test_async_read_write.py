# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


class TestAsyncRead(unittest.TestCase):
    """
    async_read parameters: 
        src(pin_memory), dst(cuda), index(cpu),
        buffer(pin_memory), offset(cpu), count(cpu)
    """

    def setUp(self):
        self.empty = paddle.to_tensor(
            np.array(
                [], dtype="int64"), place=paddle.CPUPlace())
        data = np.random.randn(100, 100).astype("float32")
        self.src = paddle.to_tensor(data, place=paddle.CUDAPinnedPlace())
        self.dst = paddle.empty(shape=[100, 100])
        self.index = paddle.to_tensor(
            np.array(
                [1, 3, 5, 7, 9], dtype="int64")).cpu()
        self.buffer = paddle.empty(shape=[50, 100]).pin_memory()

    def test_async_read_empty_offset_and_count(self):
        core.async_read(self.src, self.dst, self.index, self.buffer, self.empty,
                        self.empty)
        array1 = paddle.gather(self.src, self.index)
        array2 = self.dst[:len(self.index)]
        self.assertTrue(np.allclose(array1.numpy(), array2.numpy()))

    def test_async_read_success(self):
        offset = paddle.to_tensor(
            np.array(
                [10, 20], dtype="int64"), place=paddle.CPUPlace())
        count = paddle.to_tensor(
            np.array(
                [5, 10], dtype="int64"), place=paddle.CPUPlace())
        core.async_read(self.src, self.dst, self.index, self.buffer, offset,
                        count)

        # index data
        index_array1 = paddle.gather(self.src, self.index)
        count_numel = paddle.sum(count).numpy()[0]
        index_array2 = self.dst[count_numel:count_numel + len(self.index)]
        self.assertTrue(np.allclose(index_array1.numpy(), index_array2.numpy()))

        # offset, count
        offset_a = paddle.gather(self.src, paddle.to_tensor(np.arange(10, 15)))
        offset_b = paddle.gather(self.src, paddle.to_tensor(np.arange(20, 30)))
        offset_array1 = paddle.concat([offset_a, offset_b], axis=0)
        offset_array2 = self.dst[:count_numel]
        self.assertTrue(
            np.allclose(offset_array1.numpy(), offset_array2.numpy()))


class TestAsyncWrite(unittest.TestCase):
    """
    async_read parameters:
        src(cuda), dst(pin_memory), offset(cpu), count(cpu)
    """

    def setUp(self):
        self.src = paddle.rand(shape=[100, 100])
        self.dst = paddle.empty(shape=[200, 100]).pin_memory()

    def test_async_write_success(self):
        offset = paddle.to_tensor(
            np.array(
                [0, 60], dtype="int64"), place=paddle.CPUPlace())
        count = paddle.to_tensor(
            np.array(
                [40, 60], dtype="int64"), place=paddle.CPUPlace())
        core.async_write(self.src, self.dst, offset, count)

        offset_a = paddle.gather(self.dst, paddle.to_tensor(np.arange(0, 40)))
        offset_b = paddle.gather(self.dst, paddle.to_tensor(np.arange(60, 120)))
        offset_array = paddle.concat([offset_a, offset_b], axis=0)
        self.assertTrue(np.allclose(self.src.numpy(), offset_array.numpy()))


if __name__ == "__main__":
    if core.is_compiled_with_cuda():
        unittest.main()
