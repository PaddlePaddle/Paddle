# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from test_conv3d_op import conv3d_forward_naive

import paddle
import paddle.base.dygraph as dg
import paddle.nn.functional as F
from paddle import base, nn
from paddle.base import core


class Conv3DTestCase(unittest.TestCase):
    def __init__(
        self,
        methodName='runTest',
        batch_size=4,
        spatial_shape=(8, 8, 8),
        num_channels=6,
        num_filters=8,
        filter_size=3,
        padding=0,
        stride=1,
        dilation=1,
        groups=1,
        no_bias=False,
        data_format="NCDHW",
        dtype="float32",
    ):
        super().__init__(methodName)
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.num_filters = num_filters
        self.spatial_shape = spatial_shape
        self.filter_size = filter_size

        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.no_bias = no_bias
        self.data_format = data_format
        self.dtype = dtype

    def setUp(self):
        self.channel_last = self.data_format == "NDHWC"
        if self.channel_last:
            input_shape = (
                self.batch_size,
                *self.spatial_shape,
                self.num_channels,
            )
        else:
            input_shape = (
                self.batch_size,
                self.num_channels,
                *self.spatial_shape,
            )
        self.input = np.random.randn(*input_shape).astype(self.dtype)

        if isinstance(self.filter_size, int):
            filter_size = [self.filter_size] * 3
        else:
            filter_size = self.filter_size
        self.weight_shape = weight_shape = (
            self.num_filters,
            self.num_channels // self.groups,
            *filter_size,
        )
        self.weight = np.random.uniform(-1, 1, size=weight_shape).astype(
            self.dtype
        )
        if not self.no_bias:
            self.bias = np.random.uniform(
                -1, 1, size=(self.num_filters,)
            ).astype(self.dtype)
        else:
            self.bias = None

    def base_layer(self, place):
        main = base.Program()
        start = base.Program()
        with (
            base.unique_name.guard(),
            base.program_guard(main, start),
        ):
            input_shape = (
                (-1, -1, -1, -1, self.num_channels)
                if self.channel_last
                else (-1, self.num_channels, -1, -1, -1)
            )
            x_var = paddle.static.data("input", input_shape, dtype=self.dtype)
            weight_attr = paddle.nn.initializer.Assign(self.weight)
            if self.bias is None:
                bias_attr = False
            else:
                bias_attr = paddle.nn.initializer.Assign(self.bias)
            y_var = paddle.nn.Conv3D(
                in_channels=self.num_channels,
                out_channels=self.num_filters,
                kernel_size=self.filter_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                padding_mode="zeros",
                weight_attr=weight_attr,
                bias_attr=bias_attr,
                data_format=self.data_format,
            )(x_var)
        feed_dict = {"input": self.input}
        exe = base.Executor(place)
        exe.run(start)
        (y_np,) = exe.run(main, feed=feed_dict, fetch_list=[y_var])
        return y_np

    def functional(self, place):
        main = base.Program()
        start = base.Program()
        with (
            base.unique_name.guard(),
            base.program_guard(main, start),
        ):
            input_shape = (
                (-1, -1, -1, -1, self.num_channels)
                if self.channel_last
                else (-1, self.num_channels, -1, -1, -1)
            )
            x_var = paddle.static.data("input", input_shape, dtype=self.dtype)
            w_var = paddle.static.data(
                "weight", self.weight_shape, dtype=self.dtype
            )
            if not self.no_bias:
                b_var = paddle.static.data(
                    "bias", (self.num_filters,), dtype=self.dtype
                )
            else:
                b_var = None
            y_var = F.conv3d(
                x_var,
                w_var,
                b_var,
                padding=self.padding,
                stride=self.stride,
                dilation=self.dilation,
                groups=self.groups,
                data_format=self.data_format,
            )
        feed_dict = {"input": self.input, "weight": self.weight}
        if self.bias is not None:
            feed_dict["bias"] = self.bias
        exe = base.Executor(place)
        exe.run(start)
        (y_np,) = exe.run(main, feed=feed_dict, fetch_list=[y_var])
        return y_np

    def paddle_nn_layer(self):
        x_var = paddle.to_tensor(self.input)
        x_var.stop_gradient = False
        conv = nn.Conv3D(
            self.num_channels,
            self.num_filters,
            self.filter_size,
            padding=self.padding,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
            data_format=self.data_format,
        )
        conv.weight.set_value(self.weight)
        if not self.no_bias:
            conv.bias.set_value(self.bias)
        y_var = conv(x_var)
        y_var.backward()
        y_np = y_var.numpy()
        t1 = x_var.gradient()
        return y_np, t1

    def _test_pir_equivalence(self, place):
        with paddle.pir_utils.IrGuard():
            result1 = self.base_layer(place)
            result2 = self.functional(place)
        with dg.guard(place):
            result3, g1 = self.paddle_nn_layer()
        np.testing.assert_array_almost_equal(result1, result2)
        np.testing.assert_array_almost_equal(result2, result3)

    def runTest(self):
        place = base.CPUPlace()
        self._test_pir_equivalence(place)

        if base.core.is_compiled_with_cuda() or is_custom_device():
            place = get_device_place()
            self._test_pir_equivalence(place)


class Conv3DErrorTestCase(Conv3DTestCase):
    def runTest(self):
        place = base.CPUPlace()
        with (
            dg.guard(place),
            self.assertRaises(ValueError),
        ):
            self.paddle_nn_layer()


def add_cases(suite):
    suite.addTest(Conv3DTestCase(methodName='runTest'))
    suite.addTest(
        Conv3DTestCase(methodName='runTest', stride=[1, 2, 1], dilation=2)
    )
    suite.addTest(
        Conv3DTestCase(methodName='runTest', stride=2, dilation=(2, 1, 2))
    )
    suite.addTest(
        Conv3DTestCase(methodName='runTest', padding="same", no_bias=True)
    )
    suite.addTest(
        Conv3DTestCase(
            methodName='runTest', filter_size=(3, 2, 3), padding='valid'
        )
    )
    suite.addTest(Conv3DTestCase(methodName='runTest', padding=(2, 3, 1)))
    suite.addTest(
        Conv3DTestCase(methodName='runTest', padding=[1, 2, 2, 1, 2, 3])
    )
    suite.addTest(
        Conv3DTestCase(
            methodName='runTest',
            padding=[[0, 0], [0, 0], [1, 2], [2, 1], [2, 2]],
        )
    )
    suite.addTest(Conv3DTestCase(methodName='runTest', data_format="NDHWC"))
    suite.addTest(
        Conv3DTestCase(
            methodName='runTest',
            data_format="NDHWC",
            padding=[[0, 0], [1, 1], [3, 3], [2, 2], [0, 0]],
        )
    )
    suite.addTest(
        Conv3DTestCase(methodName='runTest', groups=2, padding="valid")
    )
    suite.addTest(
        Conv3DTestCase(
            methodName='runTest',
            num_filters=6,
            num_channels=3,
            groups=3,
            padding="valid",
        )
    )


def add_error_cases(suite):
    suite.addTest(
        Conv3DErrorTestCase(methodName='runTest', num_channels=5, groups=2)
    )
    suite.addTest(
        Conv3DErrorTestCase(
            methodName='runTest', num_channels=5, groups=2, padding=[-1, 1, 3]
        )
    )


def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    add_cases(suite)
    add_error_cases(suite)
    return suite


def get_places():
    places = []
    if core.is_compiled_with_xpu():
        places.append(paddle.device.XPUPlace(0))
    elif core.is_compiled_with_cuda():
        places.append(paddle.CUDAPlace(0))
    places.append(paddle.CPUPlace())
    return places


class TestConv3dAPI_Compatibility(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.places = get_places()
        self.shape_x = [2, 3, 8, 8, 8]  # NCDHW
        self.shape_w = [6, 3, 3, 3, 3]  # Co, Cin, kD, kH, kW
        self.dtype = "float32"
        self.init_data()

    def init_data(self):
        self.np_x = np.random.rand(*self.shape_x).astype(self.dtype)
        self.np_w = np.random.rand(*self.shape_w).astype(self.dtype)
        conv_param = {
            "stride": [1, 1, 1],
            "pad": [0, 0, 0],
            "dilation": [1, 1, 1],
        }
        self.np_ref_out = conv3d_forward_naive(
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
            out1 = paddle.nn.functional.conv3d(x, w)
            paddle_dygraph_out.append(out1)
            # Keywords args (kwargs) for paddle
            out2 = paddle.nn.functional.conv3d(x=x, weight=w)
            paddle_dygraph_out.append(out2)
            # Keywords args for alias compatibility
            out3 = paddle.nn.functional.conv3d(input=x, weight=w)
            paddle_dygraph_out.append(out3)
            # Combined args and kwargs
            out4 = paddle.nn.functional.conv3d(x, weight=w)
            paddle_dygraph_out.append(out4)

            # refer to test/xpu/test_conv3d_op_xpu.py
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
            out1 = paddle.nn.functional.conv3d(x, w)
            fetch_list.append(out1)
            # Keywords args (kwargs) for paddle
            out2 = paddle.nn.functional.conv3d(x=x, weight=w)
            fetch_list.append(out2)
            # Keywords args for alias compatibility
            out3 = paddle.nn.functional.conv3d(input=x, weight=w)
            fetch_list.append(out3)
            # Combined args and kwargs
            out4 = paddle.nn.functional.conv3d(x, weight=w)
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


if __name__ == '__main__':
    unittest.main()
