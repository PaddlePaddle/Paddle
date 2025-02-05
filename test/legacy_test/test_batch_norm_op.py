#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

import numpy as np
from op import Operator
from op_test import (
    OpTest,
    _set_use_system_allocator,
    convert_float_to_uint16,
    convert_uint16_to_float,
)

import paddle
from paddle import base
from paddle.base import core
from paddle.base.framework import grad_var_name

_set_use_system_allocator(True)


def _reference_testing(x, scale, offset, mean, var, epsilon, data_format):
    x_shape = x.shape
    if len(x_shape) == 2:
        if data_format == "NCHW":
            x = np.reshape(x, (x.shape[0], x.shape[1], 1, 1))
        else:
            x = np.reshape(x, (x.shape[0], 1, 1, x.shape[1]))
    if len(x_shape) == 3:
        if data_format == "NCHW":  # NCL -> NCL1
            x = np.reshape(x, (x_shape[0], x_shape[1], x_shape[2], 1))
        else:  # NLC -> NL1C
            x = np.reshape(x, (x_shape[0], x_shape[1], 1, x_shape[2]))

    if data_format == "NCHW":
        n, c, h, w = x.shape
        mean_tile = np.reshape(mean, (1, c, 1, 1))
        mean_tile = np.tile(mean_tile, (n, 1, h, w))
        var_tile = np.reshape(var, (1, c, 1, 1))
        var_tile = np.tile(var_tile, (n, 1, h, w))
        normalized = (x - mean_tile) / np.sqrt(var_tile + epsilon)
        scale_tile = np.reshape(scale, (1, c, 1, 1))
        scale_tile = np.tile(scale_tile, (n, 1, h, w))
        offset_tile = np.reshape(offset, (1, c, 1, 1))
        offset_tile = np.reshape(offset_tile, (1, c, 1, 1))
        y = normalized * scale_tile + offset_tile
    elif data_format == "NHWC":
        normalized = (x - mean) / np.sqrt(var + epsilon)
        y = normalized * scale + offset
    else:
        raise ValueError("Unknown data order.")

    if len(x_shape) == 2 or len(x_shape) == 3:
        y = np.reshape(y, x_shape)
    return y


def _cal_mean_variance(x, epsilon, data_format):
    assert data_format in ['NCHW', 'NHWC']
    x_shape = x.shape
    if len(x_shape) == 3:
        if data_format == "NCHW":  # NCL -> NCL1
            x = np.reshape(x, (x_shape[0], x_shape[1], x_shape[2], 1))
        else:  # NLC -> NL1C
            x = np.reshape(x, (x_shape[0], x_shape[1], 1, x_shape[2]))
    x_square = x * x
    axis = (0, 2, 3) if data_format == 'NCHW' else (0, 1, 2)
    C = x.shape[1] if data_format == 'NCHW' else x.shape[-1]
    x_square_sum = np.sum(x_square, axis)
    x_sum = np.sum(x, axis=axis)
    element_count = np.size(x) / C
    mean = x_sum / element_count
    var = x_square_sum / element_count - mean * mean
    return mean, var


def _reference_training(x, scale, offset, epsilon, data_format):
    x_shape = x.shape

    if len(x_shape) == 3:
        if data_format == "NCHW":  # NCL -> NCL1
            x = np.reshape(x, (x_shape[0], x_shape[1], x_shape[2], 1))
        else:  # NLC -> NL1C
            x = np.reshape(x, (x_shape[0], x_shape[1], 1, x_shape[2]))

    if data_format == "NCHW":
        n, c, h, w = x.shape
        x_square = x * x
        x_square_sum = np.sum(x_square, (0, 2, 3))
        x_sum = np.sum(x, axis=(0, 2, 3))
        element_count = np.size(x) / int(np.shape(x)[1])
        mean = x_sum / element_count
        var = x_square_sum / element_count - mean * mean
        mean_tile = np.reshape(mean, (1, c, 1, 1))
        mean_tile = np.tile(mean_tile, (n, 1, h, w))
        var_tile = np.reshape(var, (1, c, 1, 1))
        var_tile = np.tile(var_tile, (n, 1, h, w))
        normalized = (x - mean_tile) / np.sqrt(var_tile + epsilon)
        scale_tile = np.reshape(scale, (1, c, 1, 1))
        scale_tile = np.tile(scale_tile, (n, 1, h, w))
        offset_tile = np.reshape(offset, (1, c, 1, 1))
        offset_tile = np.reshape(offset_tile, (1, c, 1, 1))
        y = normalized * scale_tile + offset_tile
    elif data_format == "NHWC":
        x_square = x * x
        x_square_sum = np.sum(x_square, (0, 1, 2))
        x_sum = np.sum(x, axis=(0, 1, 2))
        element_count = np.size(x) / int(np.shape(x)[-1])
        mean = x_sum / element_count
        var = x_square_sum / element_count - mean * mean
        normalized = (x - mean) / np.sqrt(var + epsilon)
        y = normalized * scale + offset
    else:
        raise ValueError("Unknown data order.")

    if len(x_shape) == 3:
        y = np.reshape(y, x_shape)
    return y, mean, var


def _reference_grad(x, y_grad, scale, mean, var, epsilon, data_format):
    # Use the following formulas to calculate gradients:
    # grad_scale =
    #   sum(grad_y * (x - mean)) * rsqrt(var + epsilon)
    #
    # grad_offset = sum(output_y)
    #
    # x_grad =
    #   1/N * scale * rsqrt(var + epsilon) * (N * grad_y - sum(grad_y) -
    #   (x - mean) * sum(grad_y * (x - mean)) / (var + epsilon))

    # transfer from (N, C, H, W) to (N, H, W, C) to simplify computation
    if data_format != "NCHW" and data_format != "NHWC":
        raise ValueError("Unknown data order.")

    x_shape = x.shape
    if len(x_shape) == 3:
        if data_format == "NCHW":  # NCL -> NCL1
            x = np.reshape(x, (x_shape[0], x_shape[1], x_shape[2], 1))
            y_grad = np.reshape(y_grad, (x_shape[0], x_shape[1], x_shape[2], 1))
        else:  # NLC -> NL1C
            x = np.reshape(x, (x_shape[0], x_shape[1], 1, x_shape[2]))
            y_grad = np.reshape(y_grad, (x_shape[0], x_shape[1], 1, x_shape[2]))

    if data_format == "NCHW":
        x = np.transpose(x, (0, 2, 3, 1))
        y_grad = np.transpose(y_grad, (0, 2, 3, 1))

    x_grad = (
        scale
        * (
            y_grad
            - np.mean(y_grad, axis=(0, 1, 2))
            - (x - mean)
            * np.mean(y_grad * (x - mean), axis=(0, 1, 2))
            / (var + epsilon)
        )
        / np.sqrt(var + epsilon)
    )
    grad_scale = np.sum(
        y_grad * (x - mean) / np.sqrt(var + epsilon), axis=(0, 1, 2)
    )
    grad_offset = np.sum(y_grad, axis=(0, 1, 2))

    # transfer back to N, C, H, W
    if data_format == "NCHW":
        x_grad = np.transpose(x_grad, (0, 3, 1, 2))
        x = np.transpose(x, (0, 3, 1, 2))
        y_grad = np.transpose(y_grad, (0, 3, 1, 2))

    if len(x_shape) == 3:
        x_grad = np.reshape(x_grad, x_shape)

    return x_grad, grad_scale, grad_offset


def create_or_get_tensor(scope, var_name, var, place):
    tensor = scope.var(var_name).get_tensor()
    if var is not None:
        assert isinstance(var, np.ndarray)
        tensor.set(var, place)
    return tensor


def set_output_grad(scope, outputs, place, feed_dict=None):
    def __set_tensor__(name, data=None):
        out_tensor = scope.find_var(name).get_tensor()
        grad_tensor = scope.var(grad_var_name(name)).get_tensor()
        out_dtype = out_tensor.dtype()
        if data is None:
            if out_dtype == paddle.float64:
                data = np.ones(out_tensor.shape(), dtype=np.float64)
            elif out_dtype == paddle.float32:
                data = np.ones(out_tensor.shape(), dtype=np.float32)
            else:
                raise ValueError("Not supported data type " + str(out_dtype))
        grad_tensor.set(data, place)

    for output in outputs:
        data = None
        if output in feed_dict:
            data = feed_dict[output]
        __set_tensor__(output, data)


class TestBatchNormOpInference(unittest.TestCase):
    def setUp(self):
        self.dtype = np.float32
        self.use_mkldnn = False
        self.fuse_with_relu = False
        self.init_kernel_type()

    def __assert_close(self, tensor, np_array, msg, atol=1e-4):
        np.testing.assert_allclose(
            np.array(tensor), np_array, rtol=1e-05, atol=atol, err_msg=msg
        )

    def check_with_place(self, place, data_layout, dtype, shape):
        epsilon = 0.00001
        if len(shape) == 2:
            x_shape = shape
            c = x_shape[1]
        else:
            n, h, w, c = shape[0], shape[1], shape[2], shape[3]
            if data_layout == "NHWC":
                x_shape = [n, h, w, c]
            elif data_layout == "NCHW":
                x_shape = [n, c, h, w]
            else:
                raise ValueError("Unknown data layout.")
        scale_shape = [c]

        if dtype == np.uint16:
            x_val = np.random.random_sample(x_shape).astype(np.float32)
        else:
            x_val = np.random.random_sample(x_shape).astype(dtype)
        # generate some negative values to test case with relu fused
        x_val = x_val - 0.5
        scale_val = np.random.random_sample(scale_shape).astype(np.float32)
        bias_val = np.random.random_sample(scale_shape).astype(np.float32)

        mean = np.zeros(scale_shape).astype(np.float32)
        variance = np.ones(scale_shape).astype(np.float32)

        if dtype == np.uint16:
            y_out = _reference_testing(
                x_val, scale_val, bias_val, mean, variance, epsilon, data_layout
            ).astype(np.float32)
            y_out = convert_float_to_uint16(y_out)
        else:
            y_out = _reference_testing(
                x_val, scale_val, bias_val, mean, variance, epsilon, data_layout
            ).astype(dtype)
        if self.fuse_with_relu:
            y_out = np.maximum(y_out, 0)

        if dtype == np.uint16:
            x_val = convert_float_to_uint16(x_val)
        scope = core.Scope()

        # create input
        x_tensor = create_or_get_tensor(
            scope, "x_val", OpTest.np_dtype_to_base_dtype(x_val), place
        )
        scale_tensor = create_or_get_tensor(
            scope, "scale_val", OpTest.np_dtype_to_base_dtype(scale_val), place
        )
        bias_tensor = create_or_get_tensor(
            scope, "bias_val", OpTest.np_dtype_to_base_dtype(bias_val), place
        )
        mean_tensor = create_or_get_tensor(
            scope, "mean", OpTest.np_dtype_to_base_dtype(mean), place
        )
        variance_tensor = create_or_get_tensor(
            scope, "variance", OpTest.np_dtype_to_base_dtype(variance), place
        )

        # create output
        y_tensor = create_or_get_tensor(scope, "y_out", None, place)
        saved_mean_tensor = create_or_get_tensor(
            scope, "saved_mean", None, place
        )
        saved_variance_tensor = create_or_get_tensor(
            scope, "saved_variance", None, place
        )
        mean_out_tensor = mean_tensor
        variance_out_tensor = variance_tensor

        batch_norm_op = Operator(
            "batch_norm",
            # inputs
            X="x_val",
            Scale="scale_val",
            Bias="bias_val",
            Mean="mean",
            Variance="variance",
            # outputs
            Y="y_out",
            MeanOut="mean",
            VarianceOut="variance",
            SavedMean="saved_mean",
            SavedVariance="saved_variance",
            # attrs
            is_test=True,
            data_layout=data_layout,
            use_mkldnn=self.use_mkldnn,
            fuse_with_relu=self.fuse_with_relu,
            epsilon=epsilon,
        )

        batch_norm_op.run(scope, place)

        # When op is called without Executor then
        # MKL-DNN Tensor is returned. For NHWC data layout
        # dims will be in NCHW order as it is MKL-DNN way
        # of memory descripting. So we need to convert NCHW
        # dims into NHWC.
        if data_layout == "NHWC" and self.use_mkldnn:
            # Create executor to have MKL-DNN cache
            # cleared after NHWC unit test
            place = core.CPUPlace()
            exe = base.Executor(place)
            dims = y_tensor.shape()
            c = dims.pop(1)
            dims.append(c)
            y_tensor._set_dims(dims)

        # check inference result
        atol = 1e-3
        if dtype == np.uint16:
            y_tensor = convert_uint16_to_float(y_tensor)
            y_out = convert_uint16_to_float(y_out)
            atol = 1e-2
        self.__assert_close(
            y_tensor,
            y_out,
            "inference output are different at "
            + str(place)
            + ", "
            + data_layout
            + ", "
            + str(np.dtype(dtype))
            + str(np.array(y_tensor))
            + str(y_out),
            atol=atol,
        )

    def check_with_place_without_scale_and_bias(
        self, place, data_layout, dtype, shape
    ):
        epsilon = 0.00001
        if len(shape) == 2:
            x_shape = shape
            c = x_shape[1]
        else:
            n, h, w, c = shape[0], shape[1], shape[2], shape[3]
            if data_layout == "NHWC":
                x_shape = [n, h, w, c]
            elif data_layout == "NCHW":
                x_shape = [n, c, h, w]
            else:
                raise ValueError("Unknown data layout.")
        scale_shape = [c]

        if dtype == np.uint16:
            x_val = np.random.random_sample(x_shape).astype(np.float32)
        else:
            x_val = np.random.random_sample(x_shape).astype(dtype)
        # generate some negative values to test case with relu fused
        x_val = x_val - 0.5
        scale_val = np.ones(scale_shape).astype(np.float32)
        bias_val = np.zeros(scale_shape).astype(np.float32)

        mean = np.zeros(scale_shape).astype(np.float32)
        variance = np.ones(scale_shape).astype(np.float32)

        if dtype == np.uint16:
            y_out = _reference_testing(
                x_val, scale_val, bias_val, mean, variance, epsilon, data_layout
            ).astype(np.float32)
            y_out = convert_float_to_uint16(y_out)
        else:
            y_out = _reference_testing(
                x_val, scale_val, bias_val, mean, variance, epsilon, data_layout
            ).astype(dtype)
        if self.fuse_with_relu:
            y_out = np.maximum(y_out, 0)

        if dtype == np.uint16:
            x_val = convert_float_to_uint16(x_val)

        exe = paddle.static.Executor(place)
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x_ = paddle.static.data(
                name='x_val', shape=x_shape, dtype='float32'
            )
            mean_ = paddle.static.data(
                name='mean', shape=scale_shape, dtype='float32'
            )
            variance_ = paddle.static.data(
                name='variance', shape=scale_shape, dtype='float32'
            )
            y_tensor = paddle.nn.functional.batch_norm(
                x_,
                mean_,
                variance_,
                None,
                None,
                False,
                data_format=data_layout,
            )
        y_tensor = exe.run(
            main,
            feed={'x_val': x_val, 'mean': mean, 'variance': variance},
            fetch_list=[y_tensor],
        )[0]

        # check inference result
        # since op is called by Executor, there is
        # no need to transform y_tensor when data layout is "NHWC"
        atol = 1e-3
        if dtype == np.uint16:
            y_tensor = convert_uint16_to_float(y_tensor)
            y_out = convert_uint16_to_float(y_out)
            atol = 1e-2
        self.__assert_close(
            y_tensor,
            y_out,
            "inference output are different at "
            + str(place)
            + ", "
            + data_layout
            + ", "
            + str(np.dtype(dtype))
            + str(np.array(y_tensor))
            + str(y_out),
            atol=atol,
        )

    def test_check_output(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(core.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))

        for place in places:
            for data_format in ["NCHW", "NHWC"]:
                self.check_with_place(
                    place,
                    data_format,
                    self.dtype,
                    [2, 3, 4, 5],
                )
                self.check_with_place(
                    place,
                    data_format,
                    self.dtype,
                    [2, 3],
                )
                self.check_with_place_without_scale_and_bias(
                    place, data_format, self.dtype, [2, 3, 4, 5]
                )
                self.check_with_place_without_scale_and_bias(
                    place, data_format, self.dtype, [2, 3]
                )

    def init_kernel_type(self):
        pass


class TestFP16BatchNormOpInference(TestBatchNormOpInference):
    def setUp(self):
        self.dtype = np.float16
        self.use_mkldnn = False
        self.fuse_with_relu = False
        self.init_kernel_type()

    def test_check_output(self):
        places = []
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                places.append(place)
        for place in places:
            # for data_format in ["NCHW", "NHWC"]:
            for data_format in ["NCHW"]:
                self.check_with_place(
                    place,
                    data_format,
                    self.dtype,
                    [2, 3, 4, 5],
                )
                self.check_with_place(
                    place,
                    data_format,
                    self.dtype,
                    [2, 3],
                )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestBF16BatchNormOpInference(TestBatchNormOpInference):
    def setUp(self):
        self.dtype = np.uint16
        self.use_mkldnn = False
        self.fuse_with_relu = False
        self.init_kernel_type()

    def test_check_output(self):
        places = [core.CUDAPlace(0)]
        for place in places:
            # for data_format in ["NCHW", "NHWC"]:
            for data_format in ["NCHW"]:
                self.check_with_place(
                    place,
                    data_format,
                    self.dtype,
                    [2, 3, 4, 5],
                )
                self.check_with_place(
                    place,
                    data_format,
                    self.dtype,
                    [2, 3],
                )


class TestDygraphBatchNormAPIError(unittest.TestCase):

    def test_errors(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            batch_norm = paddle.nn.BatchNorm(10)
            # the input of BatchNorm must be Variable.
            x1 = base.create_lod_tensor(
                np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], base.CPUPlace()
            )
            self.assertRaises(TypeError, batch_norm, x1)

            # the input dtype of BatchNorm must be float16 or float32 or float64
            # float16 only can be set on GPU place
            x2 = paddle.static.data(
                name='x2', shape=[-1, 3, 4, 5, 6], dtype="int32"
            )
            self.assertRaises(TypeError, batch_norm, x2)


class TestDygraphBatchNormTrainableStats(unittest.TestCase):
    def test_dygraph(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            shape = [4, 10, 4, 4]

            def compute(x, is_test, trainable_statistics):
                with base.dygraph.guard(p):
                    bn = paddle.nn.BatchNorm(
                        shape[1],
                        is_test=is_test,
                        trainable_statistics=trainable_statistics,
                    )
                    y = bn(paddle.to_tensor(x))
                return y.numpy()

            x = np.random.randn(*shape).astype("float32")
            y1 = compute(x, False, False)
            y2 = compute(x, True, True)
            np.testing.assert_allclose(y1, y2, rtol=1e-05)

    def test_static(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            exe = base.Executor(p)
            shape = [4, 10, 16, 16]

            def compute(x_np, is_test, trainable_statistics):
                main_program = paddle.static.Program()
                startup_program = paddle.static.Program()
                with paddle.static.program_guard(main_program, startup_program):
                    bn = paddle.nn.BatchNorm(
                        shape[1],
                        is_test=is_test,
                        trainable_statistics=trainable_statistics,
                    )
                    x = paddle.static.data(
                        name='x', shape=x_np.shape, dtype=x_np.dtype
                    )
                    y = bn(x)
                    exe.run(startup_program)
                    r = exe.run(feed={'x': x_np}, fetch_list=[y])[0]
                return r

            x = np.random.randn(*shape).astype("float32")
            y1 = compute(x, False, False)
            y2 = compute(x, True, True)
            np.testing.assert_allclose(y1, y2, rtol=1e-05)


class TestDygraphBatchNormOpenReserveSpace(unittest.TestCase):

    def test_reservespace(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            paddle.enable_static()
            x = np.random.random(size=(3, 10, 3, 7)).astype('float32')
            x = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
            # Set this FLAG, the BatchNorm API will pass "reserve_space" argument into batch_norm op.
            os.environ['FLAGS_cudnn_batchnorm_spatial_persistent'] = '1'
            batch_norm = paddle.nn.BatchNorm(7, data_layout="NHWC")
            hidden1 = batch_norm(x)
            os.environ['FLAGS_cudnn_batchnorm_spatial_persistent'] = '0'


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
