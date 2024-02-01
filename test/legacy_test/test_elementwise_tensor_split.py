#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import unittest

import numpy as np

import paddle
from paddle.base import core


class TestElementwiseOp(unittest.TestCase):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.python_api = paddle.subtract
        self.public_python_api = paddle.subtract
        self.prim_op_type = "prim"

    def test_float16_sub(self):
        if not core.is_compiled_with_cuda():
            return

        gpu_info = paddle.device.cuda.get_device_properties()

        gpu_name = gpu_info.name
        try:
            re_result = re.split(r'[ , -]', gpu_name)
            memory = int(re_result[-1][:-2])
        except:
            memory = int(gpu_info.total_memory) // (1000**3)
        if memory < 37:  # 37GB
            return

        paddle.disable_static()
        tensor_a = paddle.rand(shape=[5120, 4, 384, 384], dtype="float16")
        tensor_b = paddle.rand(shape=[5120, 1, 384, 384], dtype="float16")
        tensor_z = paddle.subtract(tensor_a, tensor_b)

        in0, in1 = paddle.split(tensor_a, num_or_sections=2, axis=1)
        (
            out0,
            out1,
        ) = paddle.split(tensor_z, num_or_sections=2, axis=1)

        split_add0 = paddle.subtract(tensor_b, in0)
        split_add1 = paddle.subtract(tensor_b, in1)

        result1 = paddle.any(paddle.equal(out0, split_add0), [0, 1, 2, 3])
        result2 = paddle.any(paddle.equal(out1, split_add1), [0, 1, 2, 3])
        np.testing.assert_equal(result1.numpy(), True)
        np.testing.assert_equal(result2.numpy(), True)


if __name__ == '__main__':
    unittest.main()
