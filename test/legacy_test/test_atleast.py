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

import unittest

import numpy as np

import paddle


def func_ref(func, *inputs):
    return func(*inputs)


test_list = [
    (paddle.atleast_1d, np.atleast_1d),
    (paddle.atleast_2d, np.atleast_2d),
    (paddle.atleast_3d, np.atleast_3d),
]

"""
- **编程范式场景**
  常规覆盖动态图和静态图的测试场景

- **硬件场景**
  常规需覆盖 CPU、GPU 两种测试场景

- **参数组合场景**
  - 需要测试单个向量、多个向量、`(向量 ... 向量)`，等方式
  - 需要测试数字与向量混合的方式
  - 需要测试不同数据类型：float16, uint16, float32, float64, int8, int16, int32, int64, uint8, complex64, complex128, bfloat16

- **计算精度**
  需要保证前向计算的精度正确性，通过 numpy 实现的函数的对比结果

- **维度测试**
  - Paddle API 支持的最低维度为 0 维，单测中应编写相应的 0 维尺寸测试 case
  - 测试从 0 维至多维（`atleast_Nd` 中大于N）

"""


if __name__ == '__main__':
    unittest.main()
