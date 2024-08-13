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
from paddle import base
from paddle.base import core


def pixel_unshuffle_np(x, down_factor, data_format="NCHW"):
    '''Numpy implementation of pixel unshuffle'''

    if data_format == "NCHW":
        n, c, h, w = x.shape
        new_shape = (
            n,
            c,
            h // down_factor,
            down_factor,
            w // down_factor,
            down_factor,
        )
        npresult = np.reshape(x, new_shape)
        npresult = npresult.transpose(0, 1, 3, 5, 2, 4)
        oshape = [
            n,
            c * down_factor * down_factor,
            h // down_factor,
            w // down_factor,
        ]
        npresult = np.reshape(npresult, oshape)
        return npresult
    else:
        n, h, w, c = x.shape
        new_shape = (
            n,
            h // down_factor,
            down_factor,
            w // down_factor,
            down_factor,
            c,
        )
        npresult = np.reshape(x, new_shape)
        npresult = npresult.transpose(0, 1, 3, 5, 2, 4)
        oshape = [
            n,
            h // down_factor,
            w // down_factor,
            c * down_factor * down_factor,
        ]
        npresult = np.reshape(npresult, oshape)
        return npresult


def pixel_unshuffle_wrapper(x, downscale_factor, data_format):
    return paddle.nn.functional.pixel_unshuffle(
        x, downscale_factor, data_format
    )


class TestPixelUnshuffleOp(OpTest):
    '''TestPixelUnshuffleOp'''

    def setUp(self):
        '''setUp'''

        self.op_type = "pixel_unshuffle"
        self.python_api = pixel_unshuffle_wrapper
        self.init_dtype()
        self.init_data_format()
        n, c, h, w = 2, 1, 12, 12

        if self.format == "NCHW":
            shape = [n, c, h, w]
        if self.format == "NHWC":
            shape = [n, h, w, c]

        down_factor = 3

        x = np.random.random(shape).astype(self.dtype)
        npresult = pixel_unshuffle_np(x, down_factor, self.format)

        self.inputs = {"X": x}
        self.outputs = {"Out": npresult}
        self.attrs = {
            "downscale_factor": down_factor,
            "data_format": self.format,
        }

    def init_dtype(self):
        self.dtype = np.float64

    def init_data_format(self):
        '''init_data_format'''

        self.format = "NCHW"

    def test_check_output(self):
        '''test_check_output'''

        self.check_output()

    def test_check_grad(self):
        '''test_check_grad'''

        self.check_grad(["X"], "Out")


class TestChannelLast(TestPixelUnshuffleOp):
    '''TestChannelLast'''

    def init_data_format(self):
        '''init_data_format'''

        self.format = "NHWC"


class TestPixelUnshuffleFP16Op(TestPixelUnshuffleOp):
    def init_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support bfloat16",
)
class TestPixelUnshuffleBP16Op(OpTest):
    '''TestPixelUnshuffleBP16Op'''

    def setUp(self):
        self.op_type = "pixel_unshuffle"
        self.python_api = pixel_unshuffle_wrapper
        self.init_dtype()
        self.init_data_format()
        n, c, h, w = 2, 1, 12, 12

        if self.format == "NCHW":
            shape = [n, c, h, w]
        if self.format == "NHWC":
            shape = [n, h, w, c]

        down_factor = 3

        x = np.random.random(shape).astype(self.np_dtype)
        npresult = pixel_unshuffle_np(x, down_factor, self.format)

        self.inputs = {"X": x}
        self.outputs = {"Out": npresult}
        self.attrs = {
            "downscale_factor": down_factor,
            "data_format": self.format,
        }

        self.place = core.CUDAPlace(0)
        self.inputs['X'] = convert_float_to_uint16(self.inputs['X'])
        self.outputs['Out'] = convert_float_to_uint16(self.outputs['Out'])

    def init_dtype(self):
        self.dtype = np.uint16
        self.np_dtype = np.float32

    def init_data_format(self):
        self.format = "NCHW"

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place,
            ['X'],
            'Out',
        )


class TestPixelUnshuffleAPI(unittest.TestCase):
    '''TestPixelUnshuffleAPI'''

    def setUp(self):
        '''setUp'''

        self.x_1_np = np.random.random([2, 1, 12, 12]).astype("float64")
        self.x_2_np = np.random.random([2, 12, 12, 1]).astype("float64")
        self.out_1_np = pixel_unshuffle_np(self.x_1_np, 3)
        self.out_2_np = pixel_unshuffle_np(self.x_2_np, 3, "NHWC")

    def test_static_graph_functional(self):
        '''test_static_graph_functional'''

        for use_cuda in (
            [False, True] if core.is_compiled_with_cuda() else [False]
        ):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()

            paddle.enable_static()
            x_1 = paddle.static.data(
                name="x", shape=[2, 1, 12, 12], dtype="float64"
            )
            x_2 = paddle.static.data(
                name="x2", shape=[2, 12, 12, 1], dtype="float64"
            )
            out_1 = F.pixel_unshuffle(x_1, 3)
            out_2 = F.pixel_unshuffle(x_2, 3, "NHWC")

            exe = paddle.static.Executor(place=place)
            res_1, res_2 = exe.run(
                base.default_main_program(),
                feed={"x": self.x_1_np, "x2": self.x_2_np},
                fetch_list=[out_1, out_2],
                use_prune=True,
            )

            np.testing.assert_allclose(res_1, self.out_1_np, rtol=1e-05, atol=1)
            np.testing.assert_allclose(res_2, self.out_2_np, rtol=1e-05, atol=1)

    # same test between layer and functional in this op.
    def test_static_graph_layer(self):
        '''test_static_graph_layer'''

        for use_cuda in (
            [False, True] if core.is_compiled_with_cuda() else [False]
        ):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()

            paddle.enable_static()
            x_1 = paddle.static.data(
                name="x", shape=[2, 1, 12, 12], dtype="float64"
            )
            x_2 = paddle.static.data(
                name="x2", shape=[2, 12, 12, 1], dtype="float64"
            )
            # init instance
            ps_1 = paddle.nn.PixelUnshuffle(3)
            ps_2 = paddle.nn.PixelUnshuffle(3, "NHWC")
            out_1 = ps_1(x_1)
            out_2 = ps_2(x_2)
            out_1_np = pixel_unshuffle_np(self.x_1_np, 3)
            out_2_np = pixel_unshuffle_np(self.x_2_np, 3, "NHWC")

            exe = paddle.static.Executor(place=place)
            res_1, res_2 = exe.run(
                base.default_main_program(),
                feed={"x": self.x_1_np, "x2": self.x_2_np},
                fetch_list=[out_1, out_2],
                use_prune=True,
            )

            np.testing.assert_allclose(res_1, out_1_np)
            np.testing.assert_allclose(res_2, out_2_np)

    def run_dygraph(self, down_factor, data_format):
        '''run_dygraph'''

        n, c, h, w = 2, 1, 12, 12

        if data_format == "NCHW":
            shape = [n, c, h, w]
        if data_format == "NHWC":
            shape = [n, h, w, c]

        x = np.random.random(shape).astype("float64")

        npresult = pixel_unshuffle_np(x, down_factor, data_format)

        for use_cuda in (
            [False, True] if core.is_compiled_with_cuda() else [False]
        ):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()

            paddle.disable_static(place=place)

            pixel_unshuffle = paddle.nn.PixelUnshuffle(
                down_factor, data_format=data_format
            )
            result = pixel_unshuffle(paddle.to_tensor(x))

            np.testing.assert_allclose(result.numpy(), npresult, rtol=1e-05)

            result_functional = F.pixel_unshuffle(
                paddle.to_tensor(x), 3, data_format
            )
            np.testing.assert_allclose(
                result_functional.numpy(), npresult, rtol=1e-05
            )

            pixel_unshuffle_str = f'downscale_factor={down_factor}'
            if data_format != 'NCHW':
                pixel_unshuffle_str += f', data_format={data_format}'
            self.assertEqual(pixel_unshuffle.extra_repr(), pixel_unshuffle_str)

    def test_dygraph1(self):
        '''test_dygraph1'''

        self.run_dygraph(3, "NCHW")

    def test_dygraph2(self):
        '''test_dygraph2'''

        self.run_dygraph(3, "NHWC")


class TestPixelUnshuffleError(unittest.TestCase):
    '''TestPixelUnshuffleError'''

    def test_error_functional(self):
        '''test_error_functional'''

        def error_input():
            with paddle.base.dygraph.guard():
                x = np.random.random([4, 12, 12]).astype("float64")
                pixel_unshuffle = F.pixel_unshuffle(paddle.to_tensor(x), 2)

        self.assertRaises(ValueError, error_input)

        def error_downscale_factor_1():
            with paddle.base.dygraph.guard():
                x = np.random.random([2, 1, 12, 12]).astype("float64")
                pixel_unshuffle = F.pixel_unshuffle(paddle.to_tensor(x), 3.33)

        self.assertRaises(TypeError, error_downscale_factor_1)

        def error_downscale_factor_2():
            with paddle.base.dygraph.guard():
                x = np.random.random([2, 1, 12, 12]).astype("float64")
                pixel_unshuffle = F.pixel_unshuffle(paddle.to_tensor(x), -1)

        self.assertRaises(ValueError, error_downscale_factor_2)

        def error_data_format():
            with paddle.base.dygraph.guard():
                x = np.random.random([2, 1, 12, 12]).astype("float64")
                pixel_unshuffle = F.pixel_unshuffle(
                    paddle.to_tensor(x), 3, "WOW"
                )

        self.assertRaises(ValueError, error_data_format)

    def test_error_layer(self):
        '''test_error_layer'''

        def error_input_layer():
            with paddle.base.dygraph.guard():
                x = np.random.random([4, 12, 12]).astype("float64")
                ps = paddle.nn.PixelUnshuffle(2)
                ps(paddle.to_tensor(x))

        self.assertRaises(ValueError, error_input_layer)

        def error_downscale_factor_layer_1():
            with paddle.base.dygraph.guard():
                x = np.random.random([2, 1, 12, 12]).astype("float64")
                ps = paddle.nn.PixelUnshuffle(3.33)

        self.assertRaises(TypeError, error_downscale_factor_layer_1)

        def error_downscale_factor_layer_2():
            with paddle.base.dygraph.guard():
                x = np.random.random([2, 1, 12, 12]).astype("float64")
                ps = paddle.nn.PixelUnshuffle(-1)

        self.assertRaises(ValueError, error_downscale_factor_layer_2)

        def error_data_format_layer():
            with paddle.base.dygraph.guard():
                x = np.random.random([2, 1, 12, 12]).astype("float64")
                ps = paddle.nn.PixelUnshuffle(3, "MEOW")

        self.assertRaises(ValueError, error_data_format_layer)


if __name__ == "__main__":
    unittest.main()
