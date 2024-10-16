#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest

import paddle
import paddle.incubate.nn.functional as incubate_f
import paddle.nn.functional as F
from paddle.nn.layer import transformer
from paddle.nn.layer.common import Dropout, Linear
from paddle.nn.layer.norm import LayerNorm


class TestFusedFFNOp(OpTest):
    def getDtype(self):
        self.dtype = "float32"
        self.layer_norm_dtype = "float32"

    def getShape(self):
        self.batch_size = np.random.randint(1, 32)
        self.query_length = np.random.randint(32, 128)
        self.d_model = np.random.randint(32, 512)
        self.dim_feedforward = np.random.randint(32, 512)

    def getDiff(self):
        self.rtol = 1e-3
        # FIXME(limin29): Because there is a problem with the test precision
        #  on A100, atol is temporarily set to 1e-2, and it will be
        #  changed back after the precision problem is solved.
        self.atol = 1e-2
        if "V100" in paddle.device.cuda.get_device_name():
            self.atol = 1e-4

    def getActivation(self):
        self.act_method = "gelu"

    def getNormalizeBefore(self):
        self.pre_layer_norm = False

    def setUp(self):
        paddle.disable_static()
        self.__class__.op_type = "fused_feedforward"
        # check grad in test_out_and_grad()
        self.__class__.no_need_check_grad = True
        self.getDtype()
        self.getShape()
        self.getDiff()
        self.getActivation()
        self.getNormalizeBefore()
        paddle.set_default_dtype(self.dtype)
        self.weight_attr = None
        self.bias_attr = None

        self.weight_attrs = transformer._convert_param_attr_to_list(
            self.weight_attr, 2
        )
        self.bias_attrs = transformer._convert_param_attr_to_list(
            self.bias_attr, 2
        )
        self.linear1 = Linear(
            self.d_model,
            self.dim_feedforward,
            self.weight_attrs[1],
            bias_attr=self.bias_attrs[1],
        )
        self.linear2 = Linear(
            self.dim_feedforward,
            self.d_model,
            self.weight_attrs[1],
            bias_attr=self.bias_attrs[1],
        )

        paddle.set_default_dtype(self.layer_norm_dtype)
        self.norm1 = LayerNorm(self.d_model)
        self.norm2 = LayerNorm(self.d_model)
        self.dropout = Dropout(0.0, mode="upscale_in_train")
        self.dropout1 = Dropout(0.0, mode="upscale_in_train")
        self.dropout2 = Dropout(0.0, mode="upscale_in_train")
        self.activation = getattr(F, self.act_method)

        self.src = np.random.random(
            (self.batch_size, self.query_length, self.d_model)
        ).astype(self.dtype)
        self.dout = np.random.random(
            (self.batch_size, self.query_length, self.d_model)
        ).astype(self.dtype)

    def Base(self):
        paddle.disable_static()
        tensor_src = paddle.to_tensor(self.src, stop_gradient=False)
        residual = tensor_src
        if self.pre_layer_norm:
            ln1_out = self.norm1(tensor_src)
            linear2_out = self.linear2(
                self.dropout(self.activation(self.linear1(ln1_out)))
            )
            dropout2_out = residual + self.dropout2(linear2_out)
            paddle.autograd.backward(
                [dropout2_out], [paddle.to_tensor(self.dout)], True
            )
            return dropout2_out, tensor_src.grad
        else:
            linear2_out = self.linear2(
                self.dropout(self.activation(self.linear1(tensor_src)))
            )
            dropout2_out = residual + self.dropout2(linear2_out)
            dropout2_out = self.norm2(dropout2_out)
            paddle.autograd.backward(
                [dropout2_out], [paddle.to_tensor(self.dout)], True
            )
            return dropout2_out, tensor_src.grad

    def FusedFFN(self):
        paddle.disable_static()
        linear1_weight = paddle.to_tensor(
            self.linear1.weight, stop_gradient=False
        )
        linear1_bias = paddle.to_tensor(self.linear1.bias, stop_gradient=False)
        linear2_weight = paddle.to_tensor(
            self.linear2.weight, stop_gradient=False
        )
        linear2_bias = paddle.to_tensor(self.linear2.bias, stop_gradient=False)
        ln1_scale = paddle.to_tensor(self.norm1.weight, stop_gradient=False)
        ln1_bias = paddle.to_tensor(self.norm1.bias, stop_gradient=False)
        ln2_scale = paddle.to_tensor(self.norm2.weight, stop_gradient=False)
        ln2_bias = paddle.to_tensor(self.norm2.bias, stop_gradient=False)
        x = paddle.to_tensor(self.src, stop_gradient=False)
        out = incubate_f.fused_feedforward(
            x,
            linear1_weight,
            linear2_weight,
            linear1_bias,
            linear2_bias,
            ln1_scale,
            ln1_bias,
            ln2_scale,
            ln2_bias,
            0.0,
            0.0,
            activation=self.act_method,
            pre_layer_norm=self.pre_layer_norm,
        )
        paddle.autograd.backward([out], [paddle.to_tensor(self.dout)])
        return out, x.grad

    def test_out_and_grad(self):
        paddle.seed(42)
        base_out, base_grad = self.Base()
        fused_out, fused_grad = self.FusedFFN()
        np.testing.assert_allclose(
            base_out.numpy(), fused_out.numpy(), rtol=self.rtol, atol=self.atol
        )
        np.testing.assert_allclose(
            base_grad.numpy(),
            fused_grad.numpy(),
            rtol=self.rtol,
            atol=self.atol,
        )


class TestFusedFFNOpFp16(TestFusedFFNOp):
    def getDtype(self):
        self.dtype = "float16"
        self.layer_norm_dtype = "float32"

    def getDiff(self):
        self.rtol = 1e-1
        self.atol = 1e-2

    def getShape(self):
        self.batch_size = 4
        self.query_length = 32
        self.d_model = 128
        self.dim_feedforward = 256


class TestFusedFFNOpFp64(TestFusedFFNOp):
    def getDtype(self):
        self.dtype = "float64"
        self.layer_norm_dtype = "float64"


class TestFusedFFNOpActivation(TestFusedFFNOp):
    def getActivation(self):
        self.act_method = "relu"


class TestFusedFFNOpNormalizeBefore(TestFusedFFNOp):
    def getNormalizeBefore(self):
        self.pre_layer_norm = True

    def getShape(self):
        self.batch_size = 1
        self.query_length = 1
        self.d_model = 8
        self.dim_feedforward = 8


class APITestStaticFusedFFN(unittest.TestCase):
    def setUp(self):
        self.dtype = "float32"
        self.layer_norm_dtype = "float32"
        self.batch_size = 1
        self.d_model = 8
        self.dim_feedforward = 8

    def run_fused_feedforward(
        self,
        x_data,
        linear1_weight_data,
        linear1_bias_data,
        linear2_weight_data,
        linear2_bias_data,
        ln1_scale_data,
        ln1_bias_data,
        ln2_scale_data,
        ln2_bias_data,
    ):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        paddle.seed(42)
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(
                name='x',
                shape=[self.batch_size, self.d_model, self.dim_feedforward],
                dtype=self.dtype,
            )
            linear1_weight = paddle.static.data(
                name='linear1_weight',
                shape=[self.d_model, self.dim_feedforward],
                dtype=self.dtype,
            )
            linear1_bias = paddle.static.data(
                name='linear1_bias', shape=[self.dim_feedforward]
            )
            linear2_weight = paddle.static.data(
                name='linear2_weight',
                shape=[self.dim_feedforward, self.d_model],
                dtype=self.dtype,
            )
            linear2_bias = paddle.static.data(
                name='linear2_bias', shape=[self.d_model]
            )
            ln1_scale = paddle.static.data(
                name='ln1_scale', shape=[self.d_model]
            )
            ln1_bias = paddle.static.data(name='ln1_bias', shape=[self.d_model])
            ln2_scale = paddle.static.data(
                name='ln2_scale', shape=[self.d_model]
            )
            ln2_bias = paddle.static.data(name='ln2_bias', shape=[self.d_model])

            fused_out = incubate_f.fused_feedforward(
                x,
                linear1_weight,
                linear2_weight,
                linear1_bias,
                linear2_bias,
                ln1_scale,
                ln1_bias,
                ln2_scale,
                ln2_bias,
                0.0,
                0.0,
                activation="relu",
                pre_layer_norm=False,
            )

            exe = paddle.static.Executor(paddle.CUDAPlace(0))

            fetch = exe.run(
                feed={
                    'x': x_data,
                    'linear1_weight': linear1_weight_data,
                    'linear1_bias': linear1_bias_data,
                    'linear2_weight': linear2_weight_data,
                    'linear2_bias': linear2_bias_data,
                    'ln1_scale': ln1_scale_data,
                    'ln1_bias': ln1_bias_data,
                    'ln2_scale': ln2_scale_data,
                    'ln2_bias': ln2_bias_data,
                },
                fetch_list=[fused_out],
            )

        return fetch

    def run_base_ffn(
        self,
        x_data,
        linear1_weight_data,
        linear1_bias_data,
        linear2_weight_data,
        linear2_bias_data,
        ln1_scale_data,
        ln1_bias_data,
        ln2_scale_data,
        ln2_bias_data,
    ):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        paddle.seed(42)
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(
                name='x',
                shape=[self.batch_size, self.d_model, self.dim_feedforward],
                dtype=self.dtype,
            )
            linear1_weight = paddle.static.data(
                name='linear1_weight',
                shape=[self.d_model, self.dim_feedforward],
                dtype=self.dtype,
            )
            linear1_bias = paddle.static.data(
                name='linear1_bias', shape=[self.dim_feedforward]
            )
            linear2_weight = paddle.static.data(
                name='linear2_weight',
                shape=[self.dim_feedforward, self.d_model],
                dtype=self.dtype,
            )
            linear2_bias = paddle.static.data(
                name='linear2_bias', shape=[self.d_model]
            )
            ln1_scale = paddle.static.data(
                name='ln1_scale', shape=[self.d_model]
            )
            ln1_bias = paddle.static.data(name='ln1_bias', shape=[self.d_model])
            ln2_scale = paddle.static.data(
                name='ln2_scale', shape=[self.d_model]
            )
            ln2_bias = paddle.static.data(name='ln2_bias', shape=[self.d_model])

            # base ffn
            linear1_out = F.linear(x, linear1_weight, linear1_bias)
            act_out = F.relu(linear1_out)
            dropout1_out = F.dropout(x=act_out, p=0.0, training=False)
            linear2_out = F.linear(dropout1_out, linear2_weight, linear2_bias)
            dropout2_out = x + F.dropout(x=linear2_out, p=0.0, training=False)
            ln_out = F.layer_norm(
                dropout2_out,
                normalized_shape=[self.d_model],
                weight=ln2_scale,
                bias=ln2_bias,
            )

            exe = paddle.static.Executor(paddle.CUDAPlace(0))

            fetch = exe.run(
                feed={
                    'x': x_data,
                    'linear1_weight': linear1_weight_data,
                    'linear1_bias': linear1_bias_data,
                    'linear2_weight': linear2_weight_data,
                    'linear2_bias': linear2_bias_data,
                    'ln1_scale': ln1_scale_data,
                    'ln1_bias': ln1_bias_data,
                    'ln2_scale': ln2_scale_data,
                    'ln2_bias': ln2_bias_data,
                },
                fetch_list=[ln_out],
            )

        return fetch

    def test_static(self):
        paddle.enable_static()

        x_data = np.random.random(
            (self.batch_size, self.d_model, self.dim_feedforward)
        ).astype(self.dtype)
        linear1_weight_data = np.random.random(
            (self.d_model, self.dim_feedforward)
        ).astype(self.dtype)
        linear1_bias_data = np.zeros(self.dim_feedforward).astype(self.dtype)
        linear2_weight_data = np.random.random(
            (self.dim_feedforward, self.d_model)
        ).astype(self.dtype)
        linear2_bias_data = np.zeros(self.d_model).astype(self.dtype)

        ln1_scale_data = np.ones(self.d_model).astype(self.layer_norm_dtype)
        ln1_bias_data = np.zeros(self.d_model).astype(self.layer_norm_dtype)
        ln2_scale_data = np.ones(self.d_model).astype(self.layer_norm_dtype)
        ln2_bias_data = np.zeros(self.d_model).astype(self.layer_norm_dtype)

        fused_feedforward_res = self.run_fused_feedforward(
            x_data,
            linear1_weight_data,
            linear1_bias_data,
            linear2_weight_data,
            linear2_bias_data,
            ln1_scale_data,
            ln1_bias_data,
            ln2_scale_data,
            ln2_bias_data,
        )

        base_ffn_res = self.run_base_ffn(
            x_data,
            linear1_weight_data,
            linear1_bias_data,
            linear2_weight_data,
            linear2_bias_data,
            ln1_scale_data,
            ln1_bias_data,
            ln2_scale_data,
            ln2_bias_data,
        )

        np.testing.assert_allclose(
            fused_feedforward_res, base_ffn_res, rtol=1e-05, atol=0.001
        )


class TestFusedFFNOpError(unittest.TestCase):

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):

            def test_dtype():
                x = paddle.static.data(
                    name='x', shape=[1, 10, 10], dtype="int32"
                )
                linear1_weight = paddle.static.data(
                    name='linear1_weight', shape=[1, 10, 10], dtype="float32"
                )
                linear2_weight = paddle.static.data(
                    name='linear2_weight', shape=[1, 10, 10], dtype="float32"
                )
                incubate_f.fused_feedforward(x, linear1_weight, linear2_weight)

            self.assertRaises(TypeError, test_dtype)

            def test_dropout_rate_type():
                x = paddle.static.data(
                    name='x1', shape=[1, 10, 10], dtype="float32"
                )
                linear1_weight = paddle.static.data(
                    name='linear1_weight1', shape=[10, 10], dtype="float32"
                )
                linear2_weight = paddle.static.data(
                    name='linear2_weight1', shape=[10, 10], dtype="float32"
                )
                incubate_f.fused_feedforward(
                    x, linear1_weight, linear2_weight, dropout1_rate="a"
                )

            self.assertRaises(TypeError, test_dropout_rate_type)

            def test_dropout_rate_value():
                x = paddle.static.data(
                    name='x2', shape=[1, 10, 10], dtype="float32"
                )
                linear1_weight = paddle.static.data(
                    name='linear1_weight2', shape=[10, 10], dtype="float32"
                )
                linear2_weight = paddle.static.data(
                    name='linear2_weight2', shape=[10, 10], dtype="float32"
                )
                incubate_f.fused_feedforward(
                    x, linear1_weight, linear2_weight, dropout2_rate=-1
                )

            self.assertRaises(ValueError, test_dropout_rate_value)

            def test_dropout_mode():
                x = paddle.static.data(
                    name='x3', shape=[1, 10, 10], dtype="float32"
                )
                linear1_weight = paddle.static.data(
                    name='linear1_weight3', shape=[10, 10], dtype="float32"
                )
                linear2_weight = paddle.static.data(
                    name='linear2_weight3', shape=[10, 10], dtype="float32"
                )
                incubate_f.fused_feedforward(
                    x, linear1_weight, linear2_weight, mode='test'
                )

            self.assertRaises(ValueError, test_dropout_mode)


if __name__ == "__main__":
    unittest.main()
