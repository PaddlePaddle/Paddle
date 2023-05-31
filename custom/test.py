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

import time
import unittest

import numpy as np
from custom_setup_ops import custom_fused_add

import paddle


class TestFusedAdd(unittest.TestCase):
    def test_accuracy(self):
        test_num = 100
        val_range = 50000
        total_mem = 0.0
        fused_add_time = 0.0
        normal_add_time = 0.0
        shapes = []
        for i in range(test_num):
            shape = [np.random.randint(val_range), np.random.randint(val_range)]
            shapes.append(shape)

        # accuracy
        for i, shape in enumerate(shapes):
            print(f"shape:{shape}")
            x = paddle.randn(shape, dtype=paddle.float32)
            y = paddle.randn(shape, dtype=paddle.bfloat16)
            fused_add_out = custom_fused_add(x, y)
            x.add_(paddle.cast(y, paddle.float32))
            # if i % 10 == 0:
            print(
                "i:{}, fused add out:{}, normal add out:{}".format(
                    i, fused_add_out[0], x[0]
                )
            )
            np.testing.assert_equal(fused_add_out.numpy(), x.numpy())
            del x, y, fused_add_out
        return

        for i, shape in enumerate(shapes):
            numel = shape[0] * shape[1]
            x = paddle.randn(shape, dtype=paddle.float32)
            y_bf16 = paddle.randn(shape, dtype=paddle.bfloat16)
            x_mem = numel * 4 / 1024.0 // 1024.0 / 1024.0
            y_mem = numel * 2 / 1024.0 // 1024.0 / 1024.0
            total_mem += x_mem * 2 + y_mem
            print(
                "The {} time, shape:{}, numel:{}, x mem:{}G, y_mem:{}G, total_mem:{}".format(
                    i, shape, numel, x_mem, y_mem, total_mem
                )
            )
            paddle.device.cuda.synchronize()
            begin = time.time()
            out = custom_fused_add(x, y_bf16)
            paddle.device.cuda.synchronize()
            fused_add_time += time.time() - begin
            del x, y_bf16, out

        for i, shape in enumerate(shapes):
            numel = shape[0] * shape[1]
            x = paddle.randn(shape, dtype=paddle.float32)
            y_bf16 = paddle.randn(shape, dtype=paddle.bfloat16)
            x_mem = numel * 4 / 1024.0 // 1024.0 / 1024.0
            y_mem = numel * 2 / 1024.0 // 1024.0 / 1024.0
            total_mem += x_mem * 2 + y_mem
            print(
                "The {} time, shape:{}, numel:{}, x mem:{}G, y_mem:{}G, total_mem:{}".format(
                    i, shape, numel, x_mem, y_mem, total_mem
                )
            )
            paddle.device.cuda.synchronize()
            begin = time.time()
            x.add_(paddle.cast(y_bf16, paddle.float32))
            paddle.device.cuda.synchronize()
            normal_add_time += time.time() - begin
            del x, y_bf16

        print(
            "fused_add_time:{},normal_add_time:{}".format(
                fused_add_time, normal_add_time
            )
        )


if __name__ == "__main__":
    unittest.main()
