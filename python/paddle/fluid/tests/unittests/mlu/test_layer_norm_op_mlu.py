#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from operator import mul
import paddle.fluid.core as core
import paddle.fluid as fluid
import paddle.nn.functional as F
from functools import reduce
import sys

sys.path.append('..')
from op_test import _set_use_system_allocator
from paddle.fluid import Program, program_guard
from paddle.fluid.contrib.mixed_precision.fp16_utils import _keep_layer_norm_scale_bias_to_fp32
from test_layer_norm_op import _reference_layer_norm_naive, _reference_layer_norm_grad

paddle.enable_static()

np.random.random(123)

_set_use_system_allocator(True)


class TestLayerNormOp(unittest.TestCase):

    def setUp(self):
        self.use_cudnn = True
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True

    def __assert_close(self, tensor, np_array, msg, atol=1e-4):
        np.testing.assert_allclose(np.array(tensor),
                                   np_array,
                                   rtol=1e-5,
                                   atol=atol,
                                   err_msg=msg)

    def check_forward_backward(self,
                               shape,
                               begin_norm_axis,
                               has_scale=True,
                               has_bias=True,
                               y_grad_scale=1.0,
                               use_mkldnn=False):

        def test_with_place(place,
                            shape,
                            begin_norm_axis,
                            use_mkldnn=use_mkldnn):
            # attr
            epsilon = 0.00001
            x_shape = shape
            D = reduce(mul, x_shape[begin_norm_axis:len(x_shape)], 1)
            scale_shape = [D]

            np.random.seed(123)
            x = np.random.random_sample(x_shape).astype(np.float32)
            scale = np.random.random_sample(scale_shape).astype(
                np.float32) if has_scale else None
            bias = np.random.random_sample(scale_shape).astype(
                np.float32) if has_bias else None
            y_grad = (np.random.random_sample(x_shape) * y_grad_scale).astype(
                np.float32)

            # reference forward & backward
            y, mean, variance = _reference_layer_norm_naive(
                x, scale, bias, epsilon, begin_norm_axis)
            x_grad, scale_grad, bias_grad = _reference_layer_norm_grad(
                x, y_grad, scale, bias, mean, variance, begin_norm_axis)

            var_dict = locals()
            var_dict['y@GRAD'] = y_grad
            var_names = ['x', 'mean', 'variance', 'y', 'y@GRAD']
            if has_scale:
                var_names += ['scale']
            if has_bias:
                var_names += ['bias']
            ground_truth = {name: var_dict[name] for name in var_names}

            program = fluid.Program()
            with fluid.program_guard(program):
                block = program.global_block()
                for name in ground_truth:
                    block.create_var(name=name,
                                     dtype='float32',
                                     shape=ground_truth[name].shape)
                inputs = {"X": block.var('x')}
                fetch_list = [
                    'y',
                    'mean',
                    'variance',
                    'x@GRAD',
                ]
                if has_scale:
                    inputs["Scale"] = block.var('scale')
                    fetch_list += ['scale@GRAD']
                if has_bias:
                    inputs["Bias"] = block.var('bias')
                    fetch_list += ['bias@GRAD']
                layer_norm_op = block.append_op(
                    type="layer_norm",
                    inputs=inputs,
                    outputs={
                        "Y": block.var('y'),
                        "Mean": block.var('mean'),  # share the same memory
                        "Variance":
                        block.var('variance'),  # share the same memory
                    },
                    attrs={
                        "epsilon": epsilon,
                        "begin_norm_axis": begin_norm_axis,
                        "use_mkldnn": use_mkldnn
                    })
                # generate backward op_desc
                grad_op_desc_list, op_grad_to_var = core.get_grad_op_desc(
                    layer_norm_op.desc, set(), [])
                grad_op_desc = grad_op_desc_list[0]
                new_op_desc = block.desc.append_op()
                new_op_desc.copy_from(grad_op_desc)
                for var_name in grad_op_desc.output_arg_names():
                    block.desc.var(var_name.encode("ascii"))
                grad_op_desc.infer_var_type(block.desc)
                grad_op_desc.infer_shape(block.desc)
                for arg in grad_op_desc.output_arg_names():
                    grad_var = block.desc.find_var(arg.encode("ascii"))
                    grad_var.set_dtype(core.VarDesc.VarType.FP32)

                program._sync_with_cpp()
                exe = fluid.Executor(place)
                out = exe.run(program,
                              feed={
                                  name: var_dict[name]
                                  for name in ['x', 'scale', 'bias', 'y@GRAD']
                              },
                              fetch_list=fetch_list)

                self.__assert_close(y, out[0], "y")
                self.__assert_close(mean, out[1], "mean")
                self.__assert_close(1 / np.sqrt(variance), out[2], "variance",
                                    1e-3)
                self.__assert_close(x_grad, out[3], "x_grad")
                if has_scale:
                    self.__assert_close(scale_grad.reshape(-1),
                                        out[fetch_list.index('scale@GRAD')],
                                        "scale_grad", 1e-3)
                if has_bias:
                    self.__assert_close(bias_grad.reshape(-1),
                                        out[fetch_list.index('bias@GRAD')],
                                        "bias_grad")

        test_with_place(self.place, shape, begin_norm_axis)

    def test_check_forward_backward_with_scale_and_bias(self):
        self.check_forward_backward(shape=[1, 3, 4, 5], begin_norm_axis=1)
        self.check_forward_backward(shape=[2, 3, 4, 5], begin_norm_axis=1)
        self.check_forward_backward(shape=[2, 3, 4, 5],
                                    begin_norm_axis=1,
                                    has_scale=False,
                                    has_bias=True)
        self.check_forward_backward(shape=[2, 3, 4, 5],
                                    begin_norm_axis=1,
                                    has_scale=True,
                                    has_bias=False)
        self.check_forward_backward(shape=[2, 3, 4, 5],
                                    begin_norm_axis=1,
                                    has_scale=False,
                                    has_bias=False)
        self.check_forward_backward(shape=[2, 3, 4, 5], begin_norm_axis=3)
        self.check_forward_backward(shape=[92, 513, 129],
                                    begin_norm_axis=2,
                                    y_grad_scale=0.1)
        self.check_forward_backward(shape=[3, 34, 1134], begin_norm_axis=2)
        self.check_forward_backward(shape=[92, 513, 1134],
                                    begin_norm_axis=2,
                                    y_grad_scale=0.1)
        self.check_forward_backward(shape=[92, 513, 1134],
                                    begin_norm_axis=2,
                                    has_scale=False,
                                    has_bias=True,
                                    y_grad_scale=0.1)
        self.check_forward_backward(shape=[92, 513, 1134],
                                    begin_norm_axis=2,
                                    has_scale=True,
                                    has_bias=False,
                                    y_grad_scale=0.1)
        self.check_forward_backward(shape=[92, 513, 1134],
                                    begin_norm_axis=2,
                                    has_scale=False,
                                    has_bias=False,
                                    y_grad_scale=0.1)
        self.check_forward_backward(shape=[512, 1024],
                                    begin_norm_axis=1,
                                    has_scale=True,
                                    has_bias=True)


class TestLayerNormAPI(unittest.TestCase):

    def test_case(self):
        x = fluid.layers.data(name='x',
                              shape=[64, 32, 256],
                              dtype='float32',
                              append_batch_size=False)
        x = fluid.layers.layer_norm(x,
                                    scale=True,
                                    shift=True,
                                    begin_norm_axis=1,
                                    epsilon=1e-05,
                                    param_attr=None,
                                    bias_attr=None)
        x = fluid.layers.layer_norm(x,
                                    scale=False,
                                    shift=False,
                                    begin_norm_axis=1,
                                    epsilon=1e-05,
                                    param_attr=None,
                                    bias_attr=None)
        x = fluid.layers.layer_norm(x,
                                    scale=False,
                                    shift=False,
                                    begin_norm_axis=1,
                                    epsilon=1e-05,
                                    param_attr="scale",
                                    bias_attr="shift")


class TestDygraphLayerNormAPIError(unittest.TestCase):

    def test_errors(self):
        with program_guard(Program(), Program()):
            paddle.enable_static()

            layer_norm = fluid.LayerNorm([32, 32])
            # the input of LayerNorm must be Variable.
            x1 = np.random.random((3, 32, 32)).astype('float32')
            self.assertRaises(TypeError, layer_norm, x1)

            # the input dtype of LayerNorm must be float32 or float16
            x2 = fluid.layers.data(name='x2', shape=[3, 32, 32], dtype="int32")
            self.assertRaises(TypeError, layer_norm, x2)


class TestFP16ScaleBiasLayerNorm(unittest.TestCase):

    def check_main(self, x_np, weight_np, bias_np, dtype):
        paddle.disable_static()

        weight_np = weight_np.astype(dtype)
        bias_np = bias_np.astype(dtype)

        x = paddle.to_tensor(x_np)
        weight = paddle.to_tensor(weight_np)
        bias = paddle.to_tensor(bias_np)
        x.stop_gradient = False
        weight.stop_gradient = False
        bias.stop_gradient = False
        y = F.layer_norm(x, x.shape[1:], weight, bias)
        x_g, w_g, b_g = paddle.grad(y, [x, weight, bias])
        y_np = y.numpy().astype('float32')
        x_g_np = x_g.numpy().astype('float32')
        w_g_np = w_g.numpy().astype('float16')
        b_g_np = b_g.numpy().astype('float32')

        paddle.enable_static()
        return y_np, x_g_np, w_g_np, b_g_np

    def test_main(self):
        x_np = np.random.random([10, 20]).astype('float16')
        weight_np = np.random.random([20]).astype('float16')
        bias_np = np.random.random([20]).astype('float16')

        y_np_1, x_g_np_1, w_g_np_1, b_g_np_1 = self.check_main(
            x_np, weight_np, bias_np, 'float16')
        y_np_2, x_g_np_2, w_g_np_2, b_g_np_2 = self.check_main(
            x_np, weight_np, bias_np, 'float32')

        def assert_equal(x, y):
            np.testing.assert_allclose(x, y)

        assert_equal(y_np_1, y_np_2)
        assert_equal(x_g_np_1, x_g_np_2)
        assert_equal(w_g_np_1, w_g_np_2)
        assert_equal(b_g_np_1, b_g_np_2)


class TestGetSetKeepLayerNormScaleBiasFP32Flag(unittest.TestCase):

    def test_main(self):
        self.assertTrue(_keep_layer_norm_scale_bias_to_fp32())
        _keep_layer_norm_scale_bias_to_fp32(False)
        self.assertFalse(_keep_layer_norm_scale_bias_to_fp32())
        _keep_layer_norm_scale_bias_to_fp32(True)
        self.assertTrue(_keep_layer_norm_scale_bias_to_fp32())


if __name__ == '__main__':
    unittest.main()
