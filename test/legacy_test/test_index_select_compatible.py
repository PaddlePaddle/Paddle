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

import paddle
from paddle import base


def get_places():
    places = []
    if base.is_compiled_with_cuda() or is_custom_device():
        places.append(get_device_place())
    places.append(paddle.CPUPlace())
    return places


class TestIndexSelectAPI_Compatibility(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = get_places()
        self.shape = [10, 20]
        self.index_shape = [5]
        self.axis = 1
        self.dtype = "float32"
        self.init_data()

    def init_data(self):
        self.np_input = np.random.rand(*self.shape).astype(self.dtype)
        self.np_index = np.random.randint(
            0, self.shape[self.axis], self.index_shape
        ).astype("int64")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_input)
        index = paddle.to_tensor(self.np_index)
        paddle_dygraph_out = []
        # Position args (args)
        out1 = paddle.index_select(x, index, self.axis)
        paddle_dygraph_out.append(out1)
        # Key words args (kwargs) for paddle
        out2 = paddle.index_select(x=x, index=index, axis=self.axis)
        paddle_dygraph_out.append(out2)
        # Key words args for torch
        out3 = paddle.index_select(input=x, index=index, dim=self.axis)
        paddle_dygraph_out.append(out3)
        # Combined args and kwargs
        out4 = paddle.index_select(x, index, dim=self.axis)
        paddle_dygraph_out.append(out4)
        # Tensor method args
        out5 = x.index_select(index, self.axis)
        paddle_dygraph_out.append(out5)
        # Tensor method kwargs
        out6 = x.index_select(index=index, dim=self.axis)
        paddle_dygraph_out.append(out6)

        # PyTorch positional args order: (Tensor, int, Tensor)
        out7 = paddle.index_select(x, self.axis, index)
        paddle_dygraph_out.append(out7)
        out8 = paddle.index_select(x, self.axis, index=index)
        paddle_dygraph_out.append(out8)

        # Test out
        ref_out_shape = list(self.np_input.shape)
        ref_out_shape[self.axis] = len(self.np_index)
        out9 = paddle.empty(ref_out_shape, dtype=x.dtype)
        paddle.index_select(input=x, index=index, dim=self.axis, out=out9)
        paddle_dygraph_out.append(out9)

        # Numpy reference out
        ref_out = np.take(self.np_input, self.np_index, axis=self.axis)
        # Check
        for out in paddle_dygraph_out:
            np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-05)
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            index = paddle.static.data(
                name="index", shape=self.index_shape, dtype="int64"
            )
            # Position args (args)
            out1 = paddle.index_select(x, index, self.axis)
            # Key words args (kwargs) for paddle
            out2 = paddle.index_select(x=x, index=index, axis=self.axis)
            # Key words args for torch
            out3 = paddle.index_select(input=x, index=index, dim=self.axis)
            # Combined args and kwargs
            out4 = paddle.index_select(x, index, dim=self.axis)
            # Tensor method args
            out5 = x.index_select(index, self.axis)
            # Tensor method kwargs
            out6 = x.index_select(index=index, dim=self.axis)

            # PyTorch positional args order: (Tensor, int, Tensor)
            out7 = paddle.index_select(x, self.axis, index)
            out8 = paddle.index_select(x, self.axis, index=index)

            # Do not support out in static
            ref_out = np.take(self.np_input, self.np_index, axis=self.axis)
            fetch_list = [
                out1,
                out2,
                out3,
                out4,
                out5,
                out6,
                out7,
                out8,
            ]
            for place in self.places:
                exe = base.Executor(place)
                fetches = exe.run(
                    main,
                    feed={"x": self.np_input, "index": self.np_index},
                    fetch_list=fetch_list,
                )
                for out in fetches:
                    np.testing.assert_allclose(out, ref_out, rtol=1e-05)


if __name__ == "__main__":
    unittest.main()
