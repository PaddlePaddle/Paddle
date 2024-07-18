# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from pass_test import PassTest

import paddle
from paddle.base import core
from paddle.pir.core import create_parameter


class TestTRTPattern(PassTest):
    def is_program_valid(self, program):
        return True

    def build_ir_program(self):
        for bias_shape in [[1, 32, 1, 1], [32, 1, 1], [32]]:
            with paddle.pir_utils.IrGuard():
                main_prog = paddle.static.Program()
                start_prog = paddle.static.Program()
                with paddle.pir.core.program_guard(main_prog, start_prog):
                    x = paddle.static.data(
                        name='x', shape=[3, 1, 28, 28], dtype='float32'
                    )
                    conv2d = paddle.nn.Conv2D(
                        in_channels=1,
                        out_channels=32,
                        kernel_size=3,
                        padding=1,
                        data_format='NCHW',
                        bias_attr=False,
                    )
                    conv2d_transpose = paddle.nn.Conv2DTranspose(
                        in_channels=1,
                        out_channels=32,
                        kernel_size=3,
                        padding=1,
                        data_format='NCHW',
                        bias_attr=False,
                    )
                    y = create_parameter(
                        name="y",
                        shape=bias_shape,
                        dtype='float32',
                        initializer=paddle.nn.initializer.Assign(
                            np.random.random(bias_shape).astype("float32")
                        ),
                    )
                    act_op = paddle.nn.ReLU()
                    act_out = act_op(paddle.add(conv2d(x), y))
                    conv_transpose = act_op(paddle.add(conv2d_transpose(x), y))

                    add_out = paddle.add(act_out, conv_transpose)
                    pool2d = paddle.nn.MaxPool2D(
                        kernel_size=2, stride=2, padding=0
                    )
                    padding_out = pool2d(add_out)
                    batch_norm = paddle.nn.BatchNorm(32)
                    batch_norm_out = batch_norm(padding_out)
                    softmax = paddle.nn.Softmax()
                    softmax_out = softmax(batch_norm_out)
                    reshaped_out = paddle.reshape(
                        softmax_out, [softmax_out.shape[0], -1]
                    )
                    out = paddle.assign(reshaped_out)
                    self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                    self.feeds = {
                        "x": np.random.random((3, 1, 28, 28)).astype("float32"),
                    }
                    self.fetch_list = [out]
                    self.valid_op_map = {
                        "pd_op.fused_conv2d_add_act": 0,
                    }
                    return [main_prog, start_prog]

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def sample_program(self):
        yield self.build_ir_program(), False

    def test_check_output(self):
        self.check_pass_correct()


class TestMatmulScaleTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        for x_shape in [[3, 2]]:
            for w_shape in [[2, 3]]:
                for scale_bias in [1e-7]:
                    for scale_value in [2.0]:
                        for bias_after_scale in [True]:
                            with paddle.pir_utils.IrGuard():
                                main_prog = paddle.static.Program()
                                start_prog = paddle.static.Program()
                                with paddle.static.program_guard(
                                    main_prog, start_prog
                                ):
                                    x = paddle.static.data(
                                        name='x', shape=x_shape, dtype='float32'
                                    )
                                    w = paddle.static.data(
                                        name='w', shape=w_shape, dtype='float32'
                                    )
                                    out = paddle.scale(
                                        paddle.matmul(x, w),
                                        scale=scale_value,
                                        bias=scale_bias,
                                        bias_after_scale=bias_after_scale,
                                    )
                                    out = paddle.assign(out)
                                    self.pass_attr_list = [
                                        {'trt_op_marker_pass': {}}
                                    ]
                                    self.feeds = {
                                        "x": np.random.random(x_shape).astype(
                                            "float32"
                                        ),
                                        "w": np.random.random(w_shape).astype(
                                            "float32"
                                        ),
                                    }
                                    self.fetch_list = [out]
                                    self.valid_op_map = {
                                        "pd_op.conv2d": 0,
                                    }
                                    yield [main_prog, start_prog], False

    def setUp(self):
        self.places.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct()


class TestGroupNormSiluTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        for x_shape in [[2, 6, 4, 2]]:
            dtype = None
            if core.is_compiled_with_xpu():
                dtype = 'float32'
            elif core.is_compiled_with_cuda():
                dtype = 'float16'
            for epilson in [1e-5]:
                for groups in [2]:
                    rand_value = (
                        0.001
                        * paddle.rand(shape=[x_shape[1]], dtype=dtype).numpy()
                    )
                    with paddle.pir_utils.IrGuard():
                        start_prog = paddle.static.Program()
                        main_prog = paddle.static.Program()
                        with paddle.pir.core.program_guard(
                            main_prog, start_prog
                        ):
                            x = paddle.static.data(
                                name='x', shape=x_shape, dtype=dtype
                            )
                            w = create_parameter(
                                shape=[x_shape[1]],
                                dtype=dtype,
                                initializer=paddle.nn.initializer.Assign(
                                    rand_value
                                ),
                            )
                            b = create_parameter(
                                shape=[x_shape[1]],
                                dtype=dtype,
                                initializer=paddle.nn.initializer.Assign(
                                    rand_value
                                ),
                            )
                            group_norm_out = paddle.nn.functional.group_norm(
                                x,
                                num_groups=groups,
                                epsilon=epilson,
                                weight=w,
                                bias=b,
                                data_format="NCHW",
                            )
                            out = paddle.nn.functional.silu(group_norm_out)
                            out = paddle.assign(out)
                            if core.is_compiled_with_xpu():
                                self.pass_attr_list = [
                                    {'trt_op_marker_pass': {}}
                                ]
                            elif core.is_compiled_with_cuda():
                                self.pass_attr_list = [
                                    {'trt_op_marker_pass': {}}
                                ]
                            self.feeds = {
                                "x": np.random.random(x_shape).astype(dtype),
                            }
                            self.fetch_list = [out]
                            if core.is_compiled_with_xpu():
                                self.valid_op_map = {
                                    "pd_op.group_norm_silu_xpu": 0,
                                }
                            elif core.is_compiled_with_cuda():
                                self.valid_op_map = {
                                    "pd_op.add_group_norm_silu": 0,
                                }

                            yield [main_prog, start_prog], False

    def setUp(self):
        if core.is_compiled_with_xpu():
            self.places.append(paddle.XPUPlace(0))
        elif core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct()


class TestFlattenConCatTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            for x_shape in [[2, 1, 1, 19]]:
                with paddle.pir.core.program_guard(main_prog, start_prog):
                    x = paddle.static.data(
                        name='x', shape=x_shape, dtype='float32'
                    )
                    flatten = paddle.nn.Flatten(start_axis=1, stop_axis=2)
                    flatten_out = flatten(
                        paddle.transpose(x, perm=[0, 3, 1, 2])
                    )
                    out = paddle.concat([flatten_out], axis=1)
                    out = paddle.assign(out)
                    self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                    self.feeds = {
                        "x": np.random.random(x_shape).astype("float32"),
                    }
                    self.fetch_list = [out]
                    self.valid_op_map = {
                        "pd_op.fusion_transpose_flatten_concat": 0,
                    }
                    yield [main_prog, start_prog], False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct()


class TestGatherNdTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[1, 3, 4], dtype='float32'
                )
                index = paddle.static.data(
                    name='index', shape=[1, 2, 2], dtype='int32'
                )
                gather_nd_out = paddle.gather_nd(x, index)
                out = paddle.assign(gather_nd_out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x": np.random.random([1, 3, 4]).astype("float32"),
                    "index": np.random.random([1, 2, 2]).astype("int32"),
                }

                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.fusion_transpose_flatten_concat": 0,
                }
                yield [main_prog, start_prog], False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct()


class TestSliceTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[4, 5, 6], dtype='float32'
                )

                # Convert starts and ends to tensors
                axes = [0, 1, 2]
                starts = [-3, 0, 2]
                ends = [3, 2, 4]

                sliced_1 = paddle.slice(x, axes=axes, starts=starts, ends=ends)
                # print("Sliced output shape:", sliced_1.shape)

                out = paddle.assign(sliced_1)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x": np.random.random([4, 5, 6]).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.conv2d": 0,
                }
                yield [main_prog, start_prog], False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct()


class TestIndexSelectTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                image_shape = paddle.static.data(
                    name='x', shape=[1, 128, 1, 1], dtype='float32'
                )
                x = paddle.arange(
                    end=image_shape[0]
                    * image_shape[1]
                    * image_shape[2]
                    * image_shape[3]
                )
                img = paddle.reshape(x, image_shape)
                flatten_out = paddle.nn.flatten(start_axis=1, stop_axis=3)
                out = paddle.assign(flatten_out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x": np.random.random([2, 3, 4, 4]).astype("int32"),
                }

                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.fusion_transpose_flatten_concat": 0,
                }
                yield [main_prog, start_prog], False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()
