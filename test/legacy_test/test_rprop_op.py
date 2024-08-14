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

import unittest

import numpy as np
from op_test import (
    OpTest,
    convert_float_to_uint16,
)
from utils import dygraph_guard

import paddle
from paddle.base import core, in_pir_mode

paddle.enable_static()


def rprop_wrapper(
    param,
    grad,
    prev,
    learning_rate,
    master_param=None,
    learning_rate_range=np.array((1e-5, 50)).astype("float32"),
    etas=np.array((0.5, 1.2)).astype("float32"),
    multi_precision=False,
):
    paddle._C_ops.rprop_(
        param,
        grad,
        prev,
        learning_rate,
        master_param,
        learning_rate_range,
        etas,
        multi_precision,
    )


class TestRpropOp(OpTest):
    def setUp(self):
        self.op_type = "rprop"
        self.python_api = rprop_wrapper
        self.python_out_sig = ['Out']
        self.conf()
        params = np.random.random((self.h, self.w)).astype("float32")
        grads = np.random.random((self.h, self.w)).astype("float32")
        prevs = np.random.random((self.h, self.w)).astype("float32")
        learning_rates = np.random.random((self.h, self.w)).astype("float32")

        scale = 0.01
        np.subtract(params, 0.5, out=params)
        np.multiply(params, scale, out=params)
        np.subtract(grads, 0.5, out=grads)
        np.multiply(grads, scale, out=grads)
        np.subtract(prevs, 0.5, out=prevs)
        np.multiply(prevs, scale, out=prevs)
        np.multiply(learning_rates, scale, out=learning_rates)

        learning_rate_min = 0.1 * scale
        learning_rate_max = 0.9 * scale
        eta_negative = 0.5
        eta_positive = 1.2

        param_outs = params.copy()
        prev_outs = prevs.copy()
        learning_rate_outs = learning_rates.copy()

        for i, param in enumerate(params):
            grad = grads[i]
            prev = prevs[i]
            lr = learning_rate_outs[i]
            param_out = param_outs[i]
            prev_out = prev_outs[i]

            sign = np.sign(np.multiply(grad, prev))
            sign[np.greater(sign, 0)] = eta_positive
            sign[np.less(sign, 0)] = eta_negative
            sign[np.equal(sign, 0)] = 1
            np.multiply(lr, sign, out=lr)
            lr[np.less(lr, learning_rate_min)] = learning_rate_min
            lr[np.greater(lr, learning_rate_max)] = learning_rate_max

            grad = grad.copy()
            grad[np.equal(sign, eta_negative)] = 0

            learning_rate_outs[i] = lr
            param_outs[i] = np.subtract(
                param_out, np.multiply(np.sign(grad), lr)
            )
            prev_outs[i] = grad.copy()

        self.inputs = {
            "param": params,
            "grad": grads,
            "prev": prevs,
            "learning_rate": learning_rates,
            "learning_rate_range": np.array(
                (learning_rate_min, learning_rate_max)
            ).astype("float32"),
            "etas": np.array((0.5, 1.2)).astype("float32"),
        }

        self.outputs = {
            "param_out": param_outs,
            "prev_out": prev_outs,
            "learning_rate_out": learning_rate_outs,
        }

    def conf(self):
        self.h = 102
        self.w = 105

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestRpropOpCase8X(TestRpropOp):
    def conf(self):
        self.h = 10
        self.w = 64


class TestRpropV2(unittest.TestCase):
    def test_rprop_dygraph(self):
        paddle.disable_static()
        value = np.arange(26).reshape(1, 26).astype("float32")
        a = paddle.to_tensor(value)
        linear = paddle.nn.Linear(26, 5)

        rprop = paddle.optimizer.Rprop(
            learning_rate=0.01,
            parameters=linear.parameters(),
        )
        out = linear(a)
        out.backward()
        rprop.step()
        rprop.clear_gradients()

    def test_raise_error(self):
        self.assertRaises(
            ValueError, paddle.optimizer.Rprop, learning_rate=None
        )
        self.assertRaises(
            ValueError,
            paddle.optimizer.Rprop,
            learning_rate=1e-3,
            learning_rate_range=np.array((1e-2, 1e-1)).astype("float32"),
        )
        self.assertRaises(
            ValueError,
            paddle.optimizer.Rprop,
            learning_rate=1e-3,
            etas=np.array((-0.1, 1.1)).astype("float32"),
        )

    def test_rprop_group_dygraph(self):
        paddle.disable_static()
        value = np.arange(26).reshape(1, 26).astype("float32")
        a = paddle.to_tensor(value)
        linear_1 = paddle.nn.Linear(26, 5)
        linear_2 = paddle.nn.Linear(5, 3)
        rprop = paddle.optimizer.Rprop(
            learning_rate=0.01,
            parameters=[
                {'params': linear_1.parameters()},
                {
                    'params': linear_2.parameters(),
                    'learning_rate': 0.1,
                },
            ],
        )
        out = linear_1(a)
        out = linear_2(out)
        out.backward()
        rprop.step()
        rprop.clear_gradients()


class TestRpropMultiPrecision2_0(unittest.TestCase):
    def dygraph_rprop_mp(self, mp):
        paddle.disable_static()
        paddle.seed(10)
        paddle.set_device('gpu')
        input = paddle.randn((2, 2))
        model = paddle.nn.Linear(2, 2)
        optimizer = paddle.optimizer.Rprop(
            parameters=model.parameters(), multi_precision=mp
        )
        if mp:
            model = paddle.amp.decorate(models=model, level='O2')
            scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        for idx in range(5):
            if mp:
                with paddle.amp.auto_cast(level='O2'):
                    output = model(input)
                    loss = paddle.mean(output)
                scaled = scaler.scale(loss)
                scaled.backward()
                scaler.minimize(optimizer, scaled)
                optimizer.clear_grad()
            else:
                output = model(input)
                loss = paddle.mean(output)
                optimizer.step()
                optimizer.clear_grad()

        return output, model.parameters()

    def static_rprop_mp(self, mp):
        paddle.enable_static()
        paddle.seed(10)
        np.random.seed(10)
        exe = paddle.static.Executor('gpu')
        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()

        with paddle.static.program_guard(train_program, startup_program):
            if in_pir_mode():
                optimizer = paddle.optimizer.Rprop(multi_precision=mp)
                linear = paddle.nn.Linear(2, 2)

                if mp:
                    linear, optimizer = paddle.amp.decorate(
                        models=linear,
                        optimizers=optimizer,
                        level='O2',
                        dtype='float16',
                    )
            else:
                optimizer = paddle.optimizer.Rprop(multi_precision=mp)
                linear = paddle.nn.Linear(2, 2)

                if mp:
                    optimizer = paddle.static.amp.decorate(
                        optimizer,
                        init_loss_scaling=128.0,
                        use_dynamic_loss_scaling=True,
                        use_pure_fp16=True,
                        use_fp16_guard=False,
                    )

            if mp:
                data = paddle.static.data(
                    shape=[2, 2], name='X', dtype='float16'
                )
            else:
                data = paddle.static.data(
                    shape=[2, 2], name='X', dtype='float32'
                )
            if in_pir_mode():
                if mp:
                    with paddle.amp.auto_cast(
                        level='O2', dtype='float16', use_promote=True
                    ):
                        hidden = linear(data)
                else:
                    hidden = linear(data)
                loss = paddle.mean(hidden)
                optimizer.minimize(loss)
            else:
                hidden = paddle.static.nn.fc(x=data, size=10)
                loss = paddle.mean(hidden)
                optimizer.minimize(loss)
                if mp:
                    optimizer.amp_init(
                        place=paddle.CUDAPlace(0),
                        scope=paddle.static.global_scope(),
                    )
                    x = np.random.random(size=(2, 2)).astype('float16')
                else:
                    x = np.random.random(size=(2, 2)).astype('float32')

        if mp:
            optimizer.amp_init(
                place=paddle.CUDAPlace(0), scope=paddle.static.global_scope()
            )
            x = np.random.random(size=(2, 2)).astype('float16')
        else:
            x = np.random.random(size=(2, 2)).astype('float32')

        exe.run(startup_program)
        out = []
        for idx in range(5):
            if in_pir_mode():
                (loss_data,) = exe.run(
                    train_program, feed={"X": x}, fetch_list=[loss]
                )
            else:
                (loss_data,) = exe.run(
                    train_program, feed={"X": x}, fetch_list=[loss.name]
                )
            out.append(loss_data)
        return out

    def test_main(self):
        if not paddle.is_compiled_with_cuda():
            return
        "Test dygraph mode"
        output1_dy, params1_dy = self.dygraph_rprop_mp(mp=True)
        output2_dy, params2_dy = self.dygraph_rprop_mp(mp=False)
        np.testing.assert_allclose(
            output1_dy.astype('float32').numpy(),
            output2_dy.astype('float32').numpy(),
            rtol=1e-05,
            atol=0.1,
        )
        for idx in range(len(params1_dy)):
            np.testing.assert_allclose(
                params1_dy[idx].astype('float32').numpy(),
                params2_dy[idx].astype('float32').numpy(),
                rtol=1e-05,
                atol=0.1,
            )
        "Test static graph mode"
        output1_st = self.static_rprop_mp(mp=True)
        output2_st = self.static_rprop_mp(mp=False)
        for idx in range(len(output1_st)):
            np.testing.assert_allclose(
                output1_st[idx].astype('float32'),
                output2_st[idx].astype('float32'),
                rtol=1e-05,
                atol=0.1,
            )


class TestRpropSimple(unittest.TestCase):
    def setUp(self) -> None:
        self.data = np.random.random(size=(2, 2)).astype('float32')

    def run_static(self):
        with paddle.pir_utils.IrGuard():
            paddle.seed(10)
            np.random.seed(10)

            exe = paddle.static.Executor('gpu')
            train_program = paddle.static.Program()
            startup_program = paddle.static.Program()

            with paddle.static.program_guard(train_program, startup_program):
                input = paddle.static.data(
                    shape=[2, 2], name='input', dtype='float32'
                )
                model = paddle.nn.Linear(2, 2)
                output = model(input)
                loss = paddle.mean(output)

                optimizer = paddle.optimizer.Rprop()
                optimizer.minimize(loss)

            exe.run(startup_program)

            out = []
            for _ in range(5):
                (loss_data,) = exe.run(
                    train_program, feed={"input": self.data}, fetch_list=[loss]
                )
                out.append(loss_data)
            return out

    def run_dygraph(self):
        with dygraph_guard():
            paddle.seed(10)
            np.random.seed(10)

            out = []
            model = paddle.nn.Linear(2, 2)
            optimizer = paddle.optimizer.Rprop(parameters=model.parameters())
            for _ in range(5):
                output = model(paddle.to_tensor(self.data))
                loss = paddle.mean(output)
                out.append(loss.numpy())
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()

            return out

    def test_main(self):
        if not paddle.is_compiled_with_cuda():
            return
        out1 = self.run_dygraph()
        out2 = self.run_static()
        np.testing.assert_allclose(out1, out2)


@unittest.skipIf(
    not core.supports_bfloat16(), 'place does not support BF16 evaluation'
)
class TestRpropOpBF16(OpTest):
    def setUp(self):
        self.op_type = "rprop"
        self.dtype = np.uint16
        self.use_mkldnn = True
        self.conf()
        params = np.random.random((self.h, self.w)).astype("float32")
        grads = np.random.random((self.h, self.w)).astype("float32")
        prevs = np.random.random((self.h, self.w)).astype("float32")
        learning_rates = np.random.random((self.h, self.w)).astype("float32")

        scale = 0.01
        np.subtract(params, 0.5, out=params)
        np.multiply(params, scale, out=params)
        np.subtract(grads, 0.5, out=grads)
        np.multiply(grads, scale, out=grads)
        np.subtract(prevs, 0.5, out=prevs)
        np.multiply(prevs, scale, out=prevs)
        np.multiply(learning_rates, scale, out=learning_rates)

        learning_rate_min = 0.1 * scale
        learning_rate_max = 0.9 * scale
        eta_negative = 0.5
        eta_positive = 1.2

        param_outs = params.copy()
        prev_outs = prevs.copy()
        learning_rate_outs = learning_rates.copy()

        for i, param in enumerate(params):
            grad = grads[i]
            prev = prevs[i]
            lr = learning_rate_outs[i]
            param_out = param_outs[i]
            prev_out = prev_outs[i]

            sign = np.sign(np.multiply(grad, prev))
            sign[np.greater(sign, 0)] = eta_positive
            sign[np.less(sign, 0)] = eta_negative
            sign[np.equal(sign, 0)] = 1
            np.multiply(lr, sign, out=lr)
            lr[np.less(lr, learning_rate_min)] = learning_rate_min
            lr[np.greater(lr, learning_rate_max)] = learning_rate_max

            grad = grad.copy()
            grad[np.equal(sign, eta_negative)] = 0

            learning_rate_outs[i] = lr
            param_outs[i] = np.subtract(
                param_out, np.multiply(np.sign(grad), lr)
            )
            prev_outs[i] = grad.copy()

        learning_rate_range = np.array(
            (learning_rate_min, learning_rate_max)
        ).astype("float32")
        etas = np.array((0.5, 1.2)).astype("float32")

        params_bf16 = convert_float_to_uint16(params)
        grads_bf16 = convert_float_to_uint16(grads)
        prevs_bf16 = convert_float_to_uint16(prevs)
        learning_rates_bf16 = convert_float_to_uint16(learning_rates)
        learning_rate_range_bf16 = convert_float_to_uint16(learning_rate_range)
        etas_bf16 = convert_float_to_uint16(etas)

        param_outs_bf16 = convert_float_to_uint16(param_outs)
        prev_outs_bf16 = convert_float_to_uint16(prev_outs)
        learning_rate_outs_bf16 = convert_float_to_uint16(learning_rate_outs)

        self.inputs = {
            "param": params_bf16,
            "grad": grads_bf16,
            "prev": prevs_bf16,
            "learning_rate": learning_rates_bf16,
            "learning_rate_range": learning_rate_range_bf16,
            "etas": etas_bf16,
        }

        self.outputs = {
            "param_out": param_outs_bf16,
            "prev_out": prev_outs_bf16,
            "learning_rate_out": learning_rate_outs_bf16,
        }

    def conf(self):
        self.h = 102
        self.w = 105

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace(), check_dygraph=False)


if __name__ == "__main__":
    unittest.main()
