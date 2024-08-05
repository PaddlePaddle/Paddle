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
                x = paddle.static.data(name='x', shape=[3, 4], dtype='int32')
                index = paddle.static.data(
                    name='index', shape=[3], dtype='int32'
                )
                index_select_out = paddle.index_select(x, index)
                out = paddle.assign(index_select_out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x": np.random.random([3, 4]).astype("int32"),
                    "index": np.random.random([3]).astype("int32"),
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


class TestCastTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[3, 4], dtype='float32')
                cast_out = paddle.cast(x, 'bool')
                out = paddle.assign(cast_out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x": np.random.random([3, 4]).astype("float32"),
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


class TestSqueezeTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[5, 1, 10], dtype='float32'
                )
                squeeze_out = paddle.squeeze(x, axis=1)
                out = paddle.assign(squeeze_out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x": np.random.random([5, 1, 10]).astype("float32"),
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


class TestUnSqueezeTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[-1, 10], dtype='float32'
                )
                unsqueeze_out = paddle.unsqueeze(x, axis=[0, 2])
                unsqueeze_out_ = paddle.unsqueeze_(unsqueeze_out, axis=0)
                out = paddle.assign(unsqueeze_out_)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x": np.random.random([5, 10]).astype("float32"),
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


class TestSplitTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[3, 9, 5], dtype='float32'
                )
                num_or_sections = [2, 3, 4]
                axis = 1
                output0, output1, output2 = paddle.split(
                    x, num_or_sections, axis
                )
                out = paddle.assign(output0)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x": np.random.random([3, 9, 5]).astype("float32"),
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

class TestNonZeroTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[4], dtype='float32')
                out_z1_tuple = paddle.nonzero(x)
                out = paddle.assign(out_z1_tuple)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x": np.array([0.0, 1.0, 0.0, 3.0]).astype("float32"),
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

class TestSplitWithNumTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True
    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[3,9,5], dtype='int64'
                )
                num_or_sections = 3
                axis = 1
                split_out = paddle.split(x, num_or_sections=num_or_sections, axis=axis)
                out = paddle.assign(split_out[0])
                out1 = paddle.assign(split_out[1])
                out2 = paddle.assign(split_out[2])
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x": np.random.random([3,9,5]).astype("int64"),
                }

                self.fetch_list = [out,out1,out2]
                self.valid_op_map = {
                    "pd_op.fusion_transpose_flatten_concat": 0,
                }
                yield [main_prog, start_prog], False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct()

class TestGeluTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[2, 2], dtype='float32')
                m = paddle.nn.GELU()
                out = m(x)
                out = paddle.assign(out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x": np.array([[-1, 0.5], [1, 1.5]]).astype("float32"),
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

class TestGreaterEqualTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[3], dtype='float32')
                y = paddle.static.data(name='y', shape=[3], dtype='float32')
                greater_equal_out = paddle.greater_equal(x, y)
                out = paddle.assign(greater_equal_out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x" : np.array([1, 2, 3]).astype("float32"),
                    "y" : np.array([1, 3, 2]).astype("float32"),
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

class TestMultiplyTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[2,2], dtype='float32')
                y = paddle.static.data(name='y', shape=[2,2], dtype='float32')
                multiply_out = paddle.multiply(x, y)
                out = paddle.assign(multiply_out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x" : np.array([[1, 2], [3, 4]]).astype("float32"),
                    "y" : np.array([[5, 6], [7, 8]]).astype("float32"),
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

class TestBatchNormTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.arange(12, dtype="float32").reshape([2, 1, 2, 3])
                running_mean = paddle.static.data(
                    name='running_mean', shape=[1], dtype='float32'
                )
                running_variance = paddle.static.data(
                    name='running_variance', shape=[1], dtype='float32'
                )
                weight = paddle.static.data(
                    name='weight', shape=[1], dtype='float32'
                )                          
                bias = paddle.static.data(
                    name='bias', shape=[1], dtype='float32'
                )         
                batch_norm_out = paddle.nn.functional.batch_norm(x, running_mean,
                                                       running_variance, weight, bias)                                 
                out = paddle.assign(batch_norm_out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x" :  np.arange(12).reshape([2,1,2,3]).astype("float32"),
                    "running_mean" :  np.array([0]).astype("float32"),
                    "running_variance" :  np.array([1]).astype("float32"),
                    "weight" :   np.array([2]).astype("float32"),
                    "bias" :   np.array([1]).astype("float32")
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

class TestSoftmaxTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[2,3,4], dtype='float32'
                )
                softmax_out = paddle.nn.functional.softmax(x)                                 
                out = paddle.assign(softmax_out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x" :  np.array([[[2.0, 3.0, 4.0, 5.0],
                   [3.0, 4.0, 5.0, 6.0],
                   [7.0, 8.0, 8.0, 9.0]],
                  [[1.0, 2.0, 3.0, 4.0],
                   [5.0, 6.0, 7.0, 8.0],
                   [6.0, 7.0, 8.0, 9.0]]]).astype("float32"),
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

class TestReluTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[3], dtype='float32'
                )
                relu_out = paddle.nn.functional.relu(x)                                 
                out = paddle.assign(relu_out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x" :  np.array([-2, 0, 1]).astype("float32"),
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

class TestReshapeTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[2,4,6], dtype='float32'
                )
                shape_tensor = [-1, 0, 3, 2]
                relu_out = paddle.reshape(x, shape_tensor)                                 
                out = paddle.assign(relu_out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x" :  np.random.random([2,4,6]).astype("float32"),
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

class TestDropoutTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                paddle.seed(2023)
                x = paddle.static.data(
                    name='x', shape=[2,3], dtype='float32'
                )
                dropout_out =  paddle.nn.functional.dropout(x, 0.5 ,training=False)                               
                out = paddle.assign(dropout_out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x" :   np.array([[1,2,3], [4,5,6]]).astype("float32"),
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

class TestBmmTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[2,2,3], dtype='float32'
                )
                y = paddle.static.data(
                    name='y', shape=[2,3,2], dtype='float32'
                )
                bmm_out = paddle.bmm(x, y)                                
                out = paddle.assign(bmm_out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x" :   np.array([[[1.0, 1.0, 1.0],
                                       [2.0, 2.0, 2.0]],
                                      [[3.0, 3.0, 3.0],
                                       [4.0, 4.0, 4.0]]]).astype("float32"),
                    
                    "y":np.array([[[1.0, 1.0],[2.0, 2.0],[3.0, 3.0]],
                     [[4.0, 4.0],[5.0, 5.0],[6.0, 6.0]]]).astype("float32"),
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

class TestConcatTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x1 = paddle.static.data(
                    name='x1', shape=[2,3], dtype='float32'
                )
                x2 = paddle.static.data(
                    name='x2', shape=[2,3], dtype='float32'
                )
                x3 = paddle.static.data(
                    name='x3', shape=[2,2], dtype='float32'
                )
                concat_out =  paddle.concat(x=[x1, x2, x3], axis=-1)                      
                out = paddle.assign(concat_out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x1" :np.array([[1, 2, 3],
                                      [4, 5, 6]]).astype("float32"),
                    "x2":np.array([[11, 12, 13],
                                   [14, 15, 16]]).astype("float32"),
                    "x3":np.array([[21, 22],
                                [23, 24]]).astype("float32"),
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

class TestFullTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                shape = [3, 2]
                full_out = paddle.full(shape=shape, fill_value=1.)           
                out = paddle.assign(full_out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
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
  
class TestAddTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[3], dtype='float32'
                )
                y = paddle.static.data(
                    name='y', shape=[3], dtype='float32'
                )                
                add_out = paddle.add(x, y)
                out = paddle.assign(add_out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                     "x": np.array([2,3,4]).astype("float32"),
                     "y":np.array([1,5,2]).astype("float32"),
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

class TestLayer_normTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[2, 2, 2, 3], dtype='float32'
                )          
                layer_norm_out = paddle.nn.functional.layer_norm(x, x.shape[1:])
                out = paddle.assign(layer_norm_out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                     "x":  np.random.random([2, 2, 2, 3]).astype("float32")
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

class TestSiluTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[4], dtype='float32'
                )          
                Silu_out = paddle.nn.functional.silu(x)
                out = paddle.assign(Silu_out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                     "x":  np.array([[1, 2, 3, 4]]).astype("float32"),
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
  
class TestConv2dTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[2,3,8,8], dtype='float32'
                )   
                w = paddle.static.data(
                    name='w', shape=[6,3,3,3], dtype='float32'
                )          
                Conv2d_out = paddle.nn.functional.conv2d(x,w)
                out = paddle.assign(Conv2d_out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x":  np.random.random([2,3,8,8]).astype("float32"),
                    "w":  np.random.random([6,3,3,3]).astype("float32"),
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

class TestPool2dTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[1,3,32,32], dtype='float32'
                )
                AvgPool2D = paddle.nn.AvgPool2D(kernel_size=2, stride=2, padding=0) 
                AvgPool2D_output = AvgPool2D(x)
                out = paddle.assign(AvgPool2D_output)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x":  np.random.uniform(-1,1,[1, 3, 32, 32]).astype("float32"),
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

class TestConv2dTransposeTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[2,3,8,8], dtype='float32'
                )   
                w = paddle.static.data(
                    name='w', shape=[3,6,3,3], dtype='float32'
                )          
                Conv2dTranspose_out = paddle.nn.functional.conv2d_transpose(x,w)
                out = paddle.assign(Conv2dTranspose_out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x":  np.random.random([2,3,8,8]).astype("float32"),
                    "w":  np.random.random([3,6,3,3]).astype("float32"),
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

class TestDepthwiseConv2dTransposeTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[2,3,8,8], dtype='float32'
                )   
                w = paddle.static.data(
                    name='w', shape=[3,1,3,3], dtype='float32'
                )          
                Conv2dTranspose_out = paddle.nn.functional.conv2d_transpose(x,w,groups=3)
                out = paddle.assign(Conv2dTranspose_out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x":  np.random.random([2,3,8,8]).astype("float32"),
                    "w":  np.random.random([3,1,3,3]).astype("float32"),
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

class TestDeformableConvTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):

                x = paddle.static.data(name='x', shape=[8,1,28,28], dtype='float32')
                kh, kw = 3, 3
                weight = paddle.static.data(name='weight', shape=[16,1,kh,kw], dtype='float32')
                offset = paddle.static.data(name='offset', shape=[8, 2 * kh * kw, 26, 26], dtype='float32')
                deformable_conv_out = paddle.vision.ops.deform_conv2d(x, offset, weight)
                out = paddle.assign(deformable_conv_out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x":  np.random.random([8,1,28,28]).astype("float32"),
                    "weight":  np.random.random([16,1,kh,kw]).astype("float32"),
                    "offset": np.random.random([8, 2 * kh * kw, 26, 26]).astype("float32"),
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

class TestArangeTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                arange_out = paddle.arange(5)
                out = paddle.assign(arange_out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
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

class TestSignTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[4], dtype='float32')
                sign_out = paddle.sign(x=x)
                out = paddle.assign(sign_out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x":  np.array([3.0, 0.0, -2.0, 1.7]).astype("float32"),
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

class TestLogicalNotTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[4], dtype='bool')
                logical_not_out =  paddle.logical_not(x)
                out = paddle.assign(logical_not_out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x":  np.array([True, False, True, False]).astype("bool"),
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

class TestTransposeTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[2,3,4], dtype='float32')
                perm0 = [1,0,2]
                transpose_out =  paddle.transpose(x, perm0)
                out = paddle.assign(transpose_out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x":  np.array(
                        [[[ 1 ,2 , 3,  4] ,[ 5,  6,  7 , 8] ,[ 9, 10, 11, 12]],
                         [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]).astype("float32"),
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

class TestGatherTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[3,2], dtype='float32')
                index = paddle.static.data(name='index', shape=[2], dtype='int32')
                gather_out =  paddle.gather(x, index, axis=0)
                out = paddle.assign(gather_out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x":  np.array([[1,2],[3,4],[5,6]]).astype("float32"),
                    "index":np.array([0,1]).astype("int32"),
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

class TestScaleTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.arange(6).astype("float32").reshape([2, 3])
                scale_out =  paddle.scale(x, scale=2.0, bias=1.0)
                out = paddle.assign(scale_out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x":  np.arange(6).astype("float32").reshape([2, 3]),
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

class TestDepthwiseConv2dTRTPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[2,3,8,8], dtype='float32'
                )   
                w = paddle.static.data(
                    name='w', shape=[6,1,3,3], dtype='float32'
                )          
                Conv2d_out = paddle.nn.functional.conv2d(x,w,groups=3)
                out = paddle.assign(Conv2d_out)
                self.pass_attr_list = [{'trt_op_marker_pass': {}}]
                self.feeds = {
                    "x":  np.random.random([2,3,8,8]).astype("float32"),
                    "w":  np.random.random([6,1,3,3]).astype("float32"),
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
