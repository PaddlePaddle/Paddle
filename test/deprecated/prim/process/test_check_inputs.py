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
from paddle.base import core


def fn(x, shape):
    out = paddle.expand(x, shape=shape)
    return out


class TestIntarrayInput(unittest.TestCase):
    """This case is set to test int_array input process during composite rule."""

    def test_non_tensor_input(self):
        core._set_prim_all_enabled(True)
        np_data = np.random.random([3, 4]).astype("float32")
        tensor_data = paddle.to_tensor(np_data)
        net = paddle.jit.to_static(fn, full_graph=True)

        _ = net(tensor_data, shape=[2, 3, 4]).numpy()
        core._set_prim_all_enabled(False)

    def test_error_input(self):
        """In composite rules, tensor shape is not supported in int_array input"""
        core._set_prim_all_enabled(True)
        np_data = np.random.random([3, 4]).astype("float32")
        tensor_data = paddle.to_tensor(np_data)
        shape = paddle.to_tensor([2, 3, 4])
        net = paddle.jit.to_static(fn, full_graph=True)
        with self.assertRaises(NotImplementedError):
            _ = net(tensor_data, shape).numpy()
        core._set_prim_all_enabled(False)


if __name__ == '__main__':
    unittest.main()
