# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, convert_float_to_uint16

import paddle
import paddle.nn.functional as F
from paddle.base import core


def channel_shuffle_np(x, groups, data_format="NCHW"):
    if data_format == "NCHW":
        n, c, h, w = x.shape
        new_shape = (n, groups, c // groups, h, w)
        npresult = np.reshape(x, new_shape)
        npresult = npresult.transpose(0, 2, 1, 3, 4)
        oshape = [n, c, h, w]
        npresult = np.reshape(npresult, oshape)
        return npresult
    else:
        n, h, w, c = x.shape
        new_shape = (n, h, w, groups, c // groups)
        npresult = np.reshape(x, new_shape)
        npresult = npresult.transpose(0, 1, 2, 4, 3)
        oshape = [n, h, w, c]
        npresult = np.reshape(npresult, oshape)
        return npresult


class TestChannelShuffleOp(OpTest):
    def setUp(self):
        self.op_type = "channel_shuffle"
        self.init_dtype()
        self.init_data_format()
        n, c, h, w = 2, 9, 4, 4
        self.python_api = paddle.nn.functional.channel_shuffle

        if self.format == "NCHW":
            shape = [n, c, h, w]
        if self.format == "NHWC":
            shape = [n, h, w, c]

        groups = 3

        x = np.random.random(shape).astype(self.dtype)
        npresult = channel_shuffle_np(x, groups, self.format)

        self.inputs = {'X': x}
        self.outputs = {'Out': npresult}
        self.attrs = {'groups': groups, "data_format": self.format}

    def init_dtype(self):
        self.dtype = 'float64'

    def init_data_format(self):
        self.format = "NCHW"

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True)


class TestChannelLast(TestChannelShuffleOp):
    def init_data_format(self):
        self.format = "NHWC"


class TestChannelShuffleAPI(unittest.TestCase):
    def setUp(self):
        self.x_2_np = np.random.random([2, 4, 4, 9]).astype("float64")
        self.out_2_np = channel_shuffle_np(self.x_2_np, 3, "NHWC")
        self.x_1_np = np.random.random([2, 9, 4, 4]).astype("float64")
        self.out_1_np = channel_shuffle_np(self.x_1_np, 3)

    def test_static_graph_functional(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            for use_cuda in (
                [False, True] if core.is_compiled_with_cuda() else [False]
            ):
                place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()

                paddle.enable_static()
                x_1 = paddle.static.data(
                    name="x", shape=[2, 9, 4, 4], dtype="float64"
                )
                out_1 = F.channel_shuffle(x_1, 3)

                exe = paddle.static.Executor(place=place)
                res_1 = exe.run(
                    paddle.static.default_main_program(),
                    feed={"x": self.x_1_np},
                    fetch_list=out_1,
                    use_prune=True,
                )

                np.testing.assert_allclose(res_1[0], self.out_1_np)

    # same test between layer and functional in this op.

    def test_static_graph_layer(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            for use_cuda in (
                [False, True] if core.is_compiled_with_cuda() else [False]
            ):
                place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()

                paddle.enable_static()
                x_1 = paddle.static.data(
                    name="x", shape=[2, 9, 4, 4], dtype="float64"
                )
                # init instance
                ps_1 = paddle.nn.ChannelShuffle(3)
                out_1 = ps_1(x_1)
                out_1_np = channel_shuffle_np(self.x_1_np, 3)

                exe = paddle.static.Executor(place=place)
                res_1 = exe.run(
                    paddle.static.default_main_program(),
                    feed={"x": self.x_1_np},
                    fetch_list=out_1,
                    use_prune=True,
                )

                np.testing.assert_allclose(res_1[0], out_1_np)

    def test_static_graph_functional_new(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            for use_cuda in (
                [False, True] if core.is_compiled_with_cuda() else [False]
            ):
                place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()

                paddle.enable_static()
                x_2 = paddle.static.data(
                    name="x2", shape=[2, 4, 4, 9], dtype="float64"
                )
                out_2 = F.channel_shuffle(x_2, 3, "NHWC")

                exe = paddle.static.Executor(place=place)
                res_2 = exe.run(
                    paddle.static.default_main_program(),
                    feed={"x2": self.x_2_np},
                    fetch_list=out_2,
                    use_prune=True,
                )

                np.testing.assert_allclose(res_2[0], self.out_2_np)

    def test_static_graph_layer_new(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            for use_cuda in (
                [False, True] if core.is_compiled_with_cuda() else [False]
            ):
                place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()

                paddle.enable_static()
                x_2 = paddle.static.data(
                    name="x2", shape=[2, 4, 4, 9], dtype="float64"
                )
                # init instance
                ps_2 = paddle.nn.ChannelShuffle(3, "NHWC")
                out_2 = ps_2(x_2)
                out_2_np = channel_shuffle_np(self.x_2_np, 3, "NHWC")

                exe = paddle.static.Executor(place=place)

                res_2 = exe.run(
                    paddle.static.default_main_program(),
                    feed={"x2": self.x_2_np},
                    fetch_list=out_2,
                    use_prune=True,
                )

                np.testing.assert_allclose(res_2[0], out_2_np)

    def run_dygraph(self, groups, data_format):
        n, c, h, w = 2, 9, 4, 4

        if data_format == "NCHW":
            shape = [n, c, h, w]
        if data_format == "NHWC":
            shape = [n, h, w, c]

        x = np.random.random(shape).astype("float64")

        npresult = channel_shuffle_np(x, groups, data_format)

        for use_cuda in (
            [False, True] if core.is_compiled_with_cuda() else [False]
        ):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()

            paddle.disable_static(place=place)

            channel_shuffle = paddle.nn.ChannelShuffle(
                groups, data_format=data_format
            )
            result = channel_shuffle(paddle.to_tensor(x))

            np.testing.assert_allclose(result.numpy(), npresult, rtol=1e-05)

            result_functional = F.channel_shuffle(
                paddle.to_tensor(x), 3, data_format
            )
            np.testing.assert_allclose(
                result_functional.numpy(), npresult, rtol=1e-05
            )

            channel_shuffle_str = f'groups={groups}'
            if data_format != 'NCHW':
                channel_shuffle_str += f', data_format={data_format}'
            self.assertEqual(channel_shuffle.extra_repr(), channel_shuffle_str)

    def test_dygraph1(self):
        self.run_dygraph(3, "NCHW")

    def test_dygraph2(self):
        self.run_dygraph(3, "NHWC")


class TestChannelShuffleError(unittest.TestCase):

    def test_error_functional(self):
        def error_input():
            with paddle.base.dygraph.guard():
                x = np.random.random([9, 4, 4]).astype("float64")
                channel_shuffle = F.channel_shuffle(paddle.to_tensor(x), 3)

        self.assertRaises(ValueError, error_input)

        def error_groups_1():
            with paddle.base.dygraph.guard():
                x = np.random.random([2, 9, 4, 4]).astype("float64")
                channel_shuffle = F.channel_shuffle(paddle.to_tensor(x), 3.33)

        self.assertRaises(TypeError, error_groups_1)

        def error_groups_2():
            with paddle.base.dygraph.guard():
                x = np.random.random([2, 9, 4, 4]).astype("float64")
                channel_shuffle = F.channel_shuffle(paddle.to_tensor(x), -1)

        self.assertRaises(ValueError, error_groups_2)

        def error_data_format():
            with paddle.base.dygraph.guard():
                x = np.random.random([2, 9, 4, 4]).astype("float64")
                channel_shuffle = F.channel_shuffle(
                    paddle.to_tensor(x), 3, "WOW"
                )

        self.assertRaises(ValueError, error_data_format)

    def test_error_layer(self):
        def error_input_layer():
            with paddle.base.dygraph.guard():
                x = np.random.random([9, 4, 4]).astype("float64")
                cs = paddle.nn.ChannelShuffle(3)
                cs(paddle.to_tensor(x))

        self.assertRaises(ValueError, error_input_layer)

        def error_groups_layer_1():
            with paddle.base.dygraph.guard():
                x = np.random.random([2, 9, 4, 4]).astype("float64")
                cs = paddle.nn.ChannelShuffle(3.33)

        self.assertRaises(TypeError, error_groups_layer_1)

        def error_groups_layer_2():
            with paddle.base.dygraph.guard():
                x = np.random.random([2, 9, 4, 4]).astype("float64")
                cs = paddle.nn.ChannelShuffle(-1)

        self.assertRaises(ValueError, error_groups_layer_2)

        def error_data_format_layer():
            with paddle.base.dygraph.guard():
                x = np.random.random([2, 9, 4, 4]).astype("float64")
                cs = paddle.nn.ChannelShuffle(3, "MEOW")

        self.assertRaises(ValueError, error_data_format_layer)


class TestChannelShuffleFP16OP(TestChannelShuffleOp):
    def init_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestChannelShuffleBF16OP(OpTest):
    def setUp(self):
        self.op_type = "channel_shuffle"
        self.init_data_format()
        n, c, h, w = 2, 9, 4, 4
        self.python_api = paddle.nn.functional.channel_shuffle
        self.dtype = np.uint16
        self.use_mkldnn = False

        if self.format == "NCHW":
            shape = [n, c, h, w]
        if self.format == "NHWC":
            shape = [n, h, w, c]

        groups = 3

        x = np.random.random(shape).astype('float32')
        out = channel_shuffle_np(x, groups, self.format)
        self.inputs = {'X': convert_float_to_uint16(x)}
        self.attrs = {'groups': groups, "data_format": self.format}
        self.outputs = {'Out': convert_float_to_uint16(out)}

    def init_data_format(self):
        self.format = "NCHW"

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_pir=True)

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(place, ['X'], 'Out', check_pir=True)


if __name__ == '__main__':
    unittest.main()
