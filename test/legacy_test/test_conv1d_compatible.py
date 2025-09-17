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

import paddle
from paddle import base
from paddle.base import core


def conv1d_forward_naive(
    input,
    filter,
    group,
    conv_param,
    padding_algorithm="EXPLICIT",
    data_format="NCL",
):
    if padding_algorithm not in ["SAME", "VALID", "EXPLICIT"]:
        raise ValueError(
            f"Unknown Attr(padding_algorithm): '{padding_algorithm}'. "
            "It can only be 'SAME' or 'VALID'."
        )

    if data_format not in ["NCL", "NLC"]:
        raise ValueError(
            f"Unknown Attr(data_format): '{data_format}' ."
            "It can only be 'NCL' or 'NLC'."
        )

    channel_last = data_format == "NLC"
    if channel_last:
        input = np.transpose(input, [0, 2, 1])

    in_n, in_c, in_l = input.shape
    f_n, f_c, f_l = filter.shape
    out_n = in_n
    out_c = f_n
    assert f_c * group == in_c
    assert np.mod(out_c, group) == 0
    sub_out_c = out_c // group
    sub_f_n = f_n // group

    stride, pad, dilation = (
        conv_param["stride"],
        conv_param["pad"],
        conv_param["dilation"],
    )

    # update pad and dilation
    def _get_padding_with_SAME(input_shape, pool_size, pool_stride):
        padding = []
        for input_size, filter_size, stride_size in zip(
            input_shape, pool_size, pool_stride
        ):
            out_size = int((input_size + stride_size - 1) / stride_size)
            pad_sum = np.max(
                ((out_size - 1) * stride_size + filter_size - input_size, 0)
            )
            pad_0 = int(pad_sum / 2)
            pad_1 = int(pad_sum - pad_0)
            padding.append(pad_0)
            padding.append(pad_1)
        return padding

    ksize = [filter.shape[2]]  # 1D kernel size
    if padding_algorithm == "VALID":
        pad = [0, 0]
    elif padding_algorithm == "SAME":
        dilation = [1]
        input_data_shape = [input.shape[2]]  # 1D input shape
        pad = _get_padding_with_SAME(input_data_shape, ksize, stride)

    pad_l_0, pad_l_1 = pad[0], pad[0]
    if len(pad) == 2:
        pad_l_0, pad_l_1 = pad[0], pad[1]

    out_l = (
        1
        + (in_l + pad_l_0 + pad_l_1 - (dilation[0] * (f_l - 1) + 1))
        // stride[0]
    )
    out = np.zeros((out_n, out_c, out_l))

    d_block_l = dilation[0] * (f_l - 1) + 1

    input_pad = np.pad(
        input,
        ((0, 0), (0, 0), (pad_l_0, pad_l_1)),
        mode="constant",
        constant_values=0,
    )

    filter_dilation = np.zeros((f_n, f_c, d_block_l))
    filter_dilation[:, :, 0 : d_block_l : dilation[0]] = filter

    for i in range(out_l):
        for g in range(group):
            input_pad_masked = input_pad[
                :,
                g * f_c : (g + 1) * f_c,
                i * stride[0] : i * stride[0] + d_block_l,
            ]

            f_sub = filter_dilation[g * sub_f_n : (g + 1) * sub_f_n, :, :]
            # sub_f_n == sub_out_c
            for k in range(sub_out_c):
                # Multiplication of Corresponding Elements, then sum all
                out[:, g * sub_out_c + k, i] = np.sum(
                    input_pad_masked * f_sub[k, :, :], axis=(1, 2)
                )

    if channel_last:
        out = np.transpose(out, [0, 2, 1])

    return out, in_n, out_l, out_c


def get_places():
    places = []
    if core.is_compiled_with_xpu():
        places.append(paddle.device.XPUPlace(0))
    elif core.is_compiled_with_cuda():
        places.append(paddle.CUDAPlace(0))
    places.append(paddle.CPUPlace())
    return places


class TestConv1dAPI_Compatibility(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.places = get_places()
        self.shape_x = [2, 3, 16]  # NCL
        self.shape_w = [6, 3, 3]  # Co, Cin, kL
        self.dtype = "float32"
        self.init_data()

    def init_data(self):
        self.np_x = np.random.rand(*self.shape_x).astype(self.dtype)
        self.np_w = np.random.rand(*self.shape_w).astype(self.dtype)
        conv_param = {"stride": [1], "pad": [0], "dilation": [1]}
        self.np_ref_out, _, _, _ = conv1d_forward_naive(
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
            out1 = paddle.nn.functional.conv1d(x, w)
            paddle_dygraph_out.append(out1)
            # Key words args (kwargs) for paddle
            out2 = paddle.nn.functional.conv1d(x=x, weight=w)
            paddle_dygraph_out.append(out2)
            # Key words args for alias compatibility - testing x->input
            out3 = paddle.nn.functional.conv1d(input=x, weight=w)
            paddle_dygraph_out.append(out3)
            # Combined args and kwargs
            out4 = paddle.nn.functional.conv1d(x, weight=w)
            paddle_dygraph_out.append(out4)

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
            out1 = paddle.nn.functional.conv1d(x, w)
            fetch_list.append(out1)
            # Key words args (kwargs) for paddle
            out2 = paddle.nn.functional.conv1d(x=x, weight=w)
            fetch_list.append(out2)
            # Key words args for alias compatibility - testing x->input
            out3 = paddle.nn.functional.conv1d(input=x, weight=w)
            fetch_list.append(out3)
            # Combined args and kwargs
            out4 = paddle.nn.functional.conv1d(x, weight=w)
            fetch_list.append(out4)

            for place in self.places:
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
