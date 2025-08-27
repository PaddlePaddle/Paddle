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

import platform
import unittest

import paddle
from paddle.incubate.tensor.manipulation import enable_activation_offload


class MyPyLayer(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, *args):
        ctx.save_for_backward(x, args)
        return x * x / 2

    @staticmethod
    def backward(ctx, y_grad):
        x, args = ctx.saved_tensor()
        return x * y_grad


class TestMain(unittest.TestCase):
    def test_main(self):
        if paddle.is_compiled_with_rocm() or not paddle.is_compiled_with_cuda():
            return

        if platform.system().lower() == "windows":
            return

        paddle.set_flags(
            {
                "FLAGS_print_offload_info": 1,
                "FLAGS_offload_inplace_tensor": True,
            }
        )
        H = 10240
        model = paddle.nn.Linear(H, H)
        enable_activation_offload(model, enable=True, retry_times=1000)

        def func(num_loop):
            z = None
            for _ in range(num_loop):
                x = paddle.randn([H, H])
                y = model(x)
                empty_tensor = paddle.empty((0, 200))
                empty_tensor._clear_to_zero_allocation()
                tmp = MyPyLayer.apply(y, paddle.empty((0, 10)), empty_tensor)
                if z is None:
                    z = tmp
                else:
                    z *= tmp

            z.mean().backward()

        func(1)
        func(25)
        enable_activation_offload(model, enable=False)
        paddle.core.offload_cached_size()


if __name__ == "__main__":
    unittest.main()
