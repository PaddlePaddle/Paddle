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

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle import _C_ops
import unittest

paddle.disable_static()


def clear_grad(w, a):
    @paddle.no_grad()
    def warp(*_):
        assert w.grad is not None
        _C_ops.scale_(w.grad, 'scale', 0.5)
        w.clear_gradient(False)

    return warp


class TestInplaceAndClearGradient(unittest.TestCase):
    def test(self):
        paddle.set_device('cpu')

        input_data = np.ones([2, 2]).astype('float32')
        w = paddle.to_tensor(input_data, 'float32', stop_gradient=False)

        _clear_grad = clear_grad(w, a="1")
        w._register_backward_hook(_clear_grad)

        for i in range(10):
            out = _C_ops.scale(w, 'scale', 0.1)
            out.backward()


if __name__ == '__main__':
    unittest.main()
