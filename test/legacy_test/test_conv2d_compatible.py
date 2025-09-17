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
from test_conv2d_op import conv2d_forward_naive

import paddle
from paddle import base
from paddle.base import core


def get_places():
    places = []
    if core.is_compiled_with_xpu():
        places.append(paddle.device.XPUPlace(0))
    elif core.is_compiled_with_cuda():
        places.append(paddle.CUDAPlace(0))
    places.append(paddle.CPUPlace())
    return places


class TestConv2dAPI_Compatibility(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.places = get_places()
        self.shape_x = [2, 3, 16, 16]  # NCHW
        self.shape_w = [6, 3, 3, 3]  # Co, Cin, kH, kW
        self.dtype = "float32"
        self.init_data()

    def init_data(self):
        self.np_x = np.random.rand(*self.shape_x).astype(self.dtype)
        self.np_w = np.random.rand(*self.shape_w).astype(self.dtype)
        conv_param = {"stride": [1, 1], "pad": [0, 0], "dilation": [1, 1]}
        self.np_ref_out, _, _, _, _ = conv2d_forward_naive(
            self.np_x, self.np_w, 1, conv_param
        )

    def test_dygraph_Compatibility(self):
        for place in self.places:
            paddle.device.set_device(place)
            paddle.disable_static()
            x = paddle.to_tensor(self.np_x)
            w = paddle.to_tensor(self.np_w)

            paddle_dygraph_out = []
            # Position args (args)
            out1 = paddle.nn.functional.conv2d(x, w)
            paddle_dygraph_out.append(out1)
            # Key words args (kwargs) for paddle
            out2 = paddle.nn.functional.conv2d(x=x, weight=w)
            paddle_dygraph_out.append(out2)
            # Key words args for alias compatibility
            out3 = paddle.nn.functional.conv2d(input=x, weight=w)
            paddle_dygraph_out.append(out3)
            # Combined args and kwargs
            out4 = paddle.nn.functional.conv2d(x, weight=w)
            paddle_dygraph_out.append(out4)

            # refer to test/xpu/test_conv2d_op_xpu.py
            if isinstance(place, core.XPUPlace):
                rtol = 5e-3
                atol = 5e-3
            else:
                rtol = 1e-5
                atol = 0

            # Check all dygraph results against reference
            for out in paddle_dygraph_out:
                np.testing.assert_allclose(
                    self.np_ref_out, out.numpy(), rtol=rtol, atol=atol
                )
            paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()

        fetch_list = []
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with base.program_guard(main, startup):
            x = paddle.static.data(
                name="x", shape=self.shape_x, dtype=self.dtype
            )
            w = paddle.static.data(
                name="w", shape=self.shape_w, dtype=self.dtype
            )

            # Position args (args)
            out1 = paddle.nn.functional.conv2d(x, w)
            fetch_list.append(out1)
            # Key words args (kwargs) for paddle
            out2 = paddle.nn.functional.conv2d(x=x, weight=w)
            fetch_list.append(out2)
            # Key words args for alias compatibility
            out3 = paddle.nn.functional.conv2d(input=x, weight=w)
            fetch_list.append(out3)
            # Combined args and kwargs
            out4 = paddle.nn.functional.conv2d(x, weight=w)
            fetch_list.append(out4)

            for place in self.places:
                # refer to test/xpu/test_conv2d_op_xpu.py
                if isinstance(place, core.XPUPlace):
                    rtol = 5e-3
                    atol = 5e-3
                else:
                    rtol = 1e-5
                    atol = 0

                exe = base.Executor(place)
                fetches = exe.run(
                    main,
                    feed={"x": self.np_x, "w": self.np_w},
                    fetch_list=fetch_list,
                )
                for out in fetches:
                    np.testing.assert_allclose(
                        out, self.np_ref_out, rtol=rtol, atol=atol
                    )


if __name__ == "__main__":
    unittest.main()
