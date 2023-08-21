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
from paddle.autograd.backward import grad
from paddle.decomposition import decompose

# paddle.device.set_device("cpu")


paddle.enable_static()


class TestBuildModule(unittest.TestCase):
    def test_basic_network(self):
        # paddle.fluid.core._set_prim_backward_enabled(True)
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.static.data('x', [4, 4], dtype='float32')
            x.stop_gradient = False
            # y = paddle.static.data('y', [4, 4], dtype='float32')
            # y.stop_gradient = False
            # dy = paddle.static.data('dy', [4], dtype='float32')
            # dout = paddle.static.data('y', [4, 4], dtype='float32')
            # res = paddle.divide(x, y)
            # res = mean(x)
            res = paddle.mean(x, axis=-1)
            [res2] = decompose(main_program, [res])
            # breakpoint()

            gradients = grad(res2, x)
            print("gradients----------", gradients)
            print(main_program)

            exe = paddle.static.Executor()
            x_feed = np.ones([4, 4], dtype=np.float32) * 10
            y_feed = np.ones([4, 4], dtype=np.float32) * 2
            dy_feed = np.ones([4], dtype=np.float32)
            # breakpoint()
            (fwd, bwd) = exe.run(
                feed={'x': x_feed, 'y': y_feed, 'dy': dy_feed},
                fetch_list=[res2, gradients],
            )
            # breakpoint()
            print(fwd, bwd)
            # print(gradients)
            # self.assertEqual(sum_value, 5 * 4 * 4)


if __name__ == "__main__":
    unittest.main()
