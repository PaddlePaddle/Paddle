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

import os
import unittest

os.environ["FLAGS_enable_new_ir_api"] = "true"
import numpy as np

import paddle
from paddle._ir_ops import add_n, multiply

paddle.enable_static()


class TestPrimMode(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [4, 8]
        self.shape_y = [4, 8]
        self.x = np.random.random(self.shape_x).astype("float32")
        self.y = np.random.random(self.shape_y).astype("float32")

    # def base_net(self):
    #     main_program = paddle.static.Program()
    #     with paddle.static.program_guard(main_program):
    #         x = paddle.static.data('x', self.shape_x, dtype='float32')
    #         y = paddle.static.data('y', self.shape_y, dtype='float32')
    #         x.stop_gradient = False
    #         y.stop_gradient = False

    #         tmp = paddle.divide(x, y)

    #         cdf = add(tmp, paddle.tensor.full(x.shape, 1., x.dtype))
    #         res = multiply(tmp, cdf)
    #         gradients = grad(res, (x, y))

    #         print(main_program)

    #         exe = paddle.static.Executor()
    #         outs = exe.run(
    #             feed={
    #                 'x': self.x,
    #                 'y': self.y,
    #             },
    #             fetch_list=[res, gradients[0], gradients[1]],
    #         )

    #     print(outs)
    #     return outs

    def base_net(self):
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.static.data('x', self.shape_x, dtype='float32')
            y = paddle.static.data('y', self.shape_y, dtype='float32')
            x.stop_gradient = False
            y.stop_gradient = False

            # tmp = paddle.divide(x, y)

            tmp = add_n([x, y])
            res = multiply(x, tmp)
            # gradients = grad(res, (x, y))

            print(main_program)

            exe = paddle.static.Executor()
            outs = exe.run(
                feed={
                    'x': self.x,
                    'y': self.y,
                },
                fetch_list=[res],
            )

        print(outs)
        return outs

    def test_prim_forward(self):
        _ = self.base_net()


if __name__ == "__main__":
    unittest.main()
