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
import unittest

import numpy as np
from op_test import get_device_place, is_custom_device
from utils import dygraph_guard, static_guard

import paddle


class TestSwapaxesCompatibility(unittest.TestCase):
    def setUp(self):
        self.places = [paddle.CPUPlace()]
        if paddle.base.core.is_compiled_with_cuda() or is_custom_device():
            self.places.append(get_device_place())
        self.func = paddle.swapaxes
        self.init_data()

    def init_data(self):
        self.shape = [4, 5, 6]
        self.dtype = 'float32'
        self.dim0 = 0
        self.dim1 = 1
        self.perm = [1, 0, 2]

        self.np_input = np.random.rand(*self.shape).astype(self.dtype)
        self.np_out = np.transpose(self.np_input, axes=self.perm)

    def test_dygraph_compatibility(self):
        with dygraph_guard():
            for place in self.places:
                paddle.device.set_device(place)
                x = paddle.to_tensor(self.np_input)
                outs = []
                outs.append(paddle.swapaxes(x, perm=self.perm))
                outs.append(paddle.swapaxes(x=x, perm=self.perm))
                outs.append(paddle.swapaxes(input=x, perm=self.perm))
                outs.append(paddle.swapaxes(x, self.dim0, self.dim1))
                outs.append(
                    paddle.swapaxes(x=x, axis0=self.dim0, axis1=self.dim1)
                )
                outs.append(
                    paddle.swapaxes(input=x, axis0=self.dim0, axis1=self.dim1)
                )

                outs.append(x.swapaxes(self.perm))
                outs.append(x.swapaxes(self.dim0, self.dim1))
                outs.append(x.swapaxes(perm=self.perm))
                outs.append(x.swapaxes(axis0=self.dim0, axis1=self.dim1))
                outs.append(x.swapaxes(self.dim0, axis1=self.dim1))

                for out in outs:
                    np.testing.assert_array_equal(self.np_out, out.numpy())

    def test_static_compatibility(self):
        with static_guard():
            for place in self.places:
                main = paddle.static.Program()
                startup = paddle.static.Program()
                with paddle.base.program_guard(main, startup):
                    x = paddle.static.data(
                        name="x", shape=self.shape, dtype=self.dtype
                    )
                    outs = []
                    outs.append(paddle.swapaxes(x, perm=self.perm))
                    outs.append(paddle.swapaxes(x=x, perm=self.perm))
                    outs.append(paddle.swapaxes(input=x, perm=self.perm))
                    outs.append(paddle.swapaxes(x, self.dim0, self.dim1))
                    outs.append(
                        paddle.swapaxes(x=x, axis0=self.dim0, axis1=self.dim1)
                    )
                    outs.append(
                        paddle.swapaxes(
                            input=x, axis0=self.dim0, axis1=self.dim1
                        )
                    )

                    outs.append(x.swapaxes(self.perm))
                    outs.append(x.swapaxes(self.dim0, self.dim1))
                    outs.append(x.swapaxes(perm=self.perm))
                    outs.append(x.swapaxes(axis0=self.dim0, axis1=self.dim1))
                    outs.append(x.swapaxes(self.dim0, axis1=self.dim1))

                    exe = paddle.base.Executor(place)
                    fetches = exe.run(
                        main,
                        feed={"x": self.np_input},
                        fetch_list=outs,
                    )
                    for out in fetches:
                        np.testing.assert_array_equal(self.np_out, out)


if __name__ == "__main__":
    unittest.main()
