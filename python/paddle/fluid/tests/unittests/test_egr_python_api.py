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

import paddle.fluid.core as core
import paddle.fluid.eager.eager_tensor_patch_methods as eager_tensor_patch_methods
import paddle
import numpy as np
from paddle.fluid import eager_guard
import unittest


class EagerScaleTestCase(unittest.TestCase):
    def test_scale_base(self):
        with eager_guard():
            paddle.set_device("cpu")
            arr = np.ones([4, 16, 16, 32]).astype('float32')
            a = paddle.to_tensor(arr, 'float32', core.CPUPlace())
            print(arr)
            print("=============")
            print(a)
            a = core.eager.scale(a, 2.0, 0.9, True, False)
            for i in range(0, 100):
                a = core.eager.scale(a, 2.0, 0.9, True, False)
            print(a.shape)
            print(a.stop_gradient)
            a.stop_gradient = False
            print(a.stop_gradient)
            a.stop_gradient = True
            print(a.stop_gradient)
            print(a)


with eager_guard():
    paddle.set_device("cpu")
    arr = np.ones([4, 16, 16, 32]).astype('float32')
    a = paddle.to_tensor(arr, 'float32', core.CPUPlace())
    a = core.eager.scale(a, 2.0, 0.9, True, False)
    for i in range(0, 100):
        a = core.eager.scale(a, 2.0, 0.9, True, False)
    print(a.shape)
    print(a.stop_gradient)
    a.stop_gradient = False
    print(a.stop_gradient)
    a.stop_gradient = True
    print(a.stop_gradient)
    print(a)
