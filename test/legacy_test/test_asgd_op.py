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
from paddle.base import core

paddle.enable_static()


def asgd_wrapper(
    param,
    grad,
    learning_rate,
    d,
    y,
    n,
    master_param=None,
    multi_precision=False,
):
    paddle._C_ops.asgd_(
        param,
        grad,
        learning_rate,
        d,
        y,
        n,
        None,
        False,
    )


class TestASGDOpMixin:
    def setUp(self):
        self.init_basic_info()
        self.init_input()
        self.update_input_dtype()
        self.init_output()
        self.update_output_dtype()

        self.inputs = {
            "param": self.params,
            "grad": self.grads,
            "learning_rate": self.learning_rate,
            "d": self.ds,
            "y": self.ys,
            "n": self.n,
        }

        self.outputs = {
            "param_out": self.params_out,
            "d_out": self.ds_out,
            "y_out": self.ys_out,
        }

    def init_basic_info(self):
        self.op_type = "asgd"
        self.python_api = asgd_wrapper
        self.python_out_sig = ['Out']
        self.h = 102
        self.w = 105

    def init_input(self):
        self.params = np.random.random((self.h, self.w))
        self.learning_rate = np.array([0.001])
        self.n = np.array([1000])
        self.grads = np.random.random((self.h, self.w))
        self.ds = np.random.random((self.h, self.w))
        self.ys = np.random.random((self.h, self.w))

    def init_output(self):
        self.ds_out = self.ds - self.ys + self.grads
        self.ys_out = self.grads.copy()
        self.params_out = (
            self.params - (self.learning_rate / self.n) * self.ds_out
        )

    def update_input_dtype(self):
        pass

    def update_output_dtype(self):
        pass

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestASGDOp(TestASGDOpMixin, OpTest):
    pass


class TestCase1(TestASGDOp):
    def update_input_dtype(self):
        self.params = self.params.astype("float32")
        self.learning_rate = self.learning_rate.astype("float32")
        self.n = self.n.astype("float32")
        self.grads = self.grads.astype("float32")
        self.ds = self.ds.astype("float32")
        self.ys = self.ys.astype("float32")


class TestCase2(TestASGDOp):
    def update_input_dtype(self):
        self.params = self.params.astype("float16")
        self.learning_rate = self.learning_rate.astype("float16")
        self.n = self.n.astype("float16")
        self.grads = self.grads.astype("float16")
        self.ds = self.ds.astype("float16")
        self.ys = self.ys.astype("float16")

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            self.check_output_with_place(core.CUDAPlace(0), check_pir=True)


class TestCase3(TestASGDOp):
    def update_input_dtype(self):
        self.params = convert_float_to_uint16(self.params)
        self.learning_rate = convert_float_to_uint16(self.learning_rate)
        self.n = convert_float_to_uint16(self.n)
        self.grads = convert_float_to_uint16(self.grads)
        self.ds = convert_float_to_uint16(self.ds)
        self.ys = convert_float_to_uint16(self.ys)

    def update_output_dtype(self):
        self.ds_out = convert_float_to_uint16(self.ds_out)
        self.ys_out = convert_float_to_uint16(self.ys_out)
        self.params_out = convert_float_to_uint16(self.params_out)

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            self.check_output_with_place(core.CUDAPlace(0), check_pir=True)


class TestCase4(TestASGDOp):
    def init_input(self):
        self.params = np.random.random((self.h, self.w))
        self.learning_rate = np.array([0.001])
        self.n = np.array([1])
        self.grads = np.random.random((self.h, self.w))
        self.ds = np.random.random((self.h, self.w))
        self.ys = np.random.random((self.h, self.w))


class TestASGDV2(unittest.TestCase):
    def test_asgd_dygraph(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear = paddle.nn.Linear(13, 5)

        asgd = paddle.optimizer.ASGD(
            learning_rate=0.001,
            batch_num=2,
            parameters=linear.parameters(),
        )
        out = linear(a)
        out.backward()
        asgd.step()
        asgd.clear_gradients()

    def test_raise_error(self):
        self.assertRaises(
            ValueError,
            paddle.optimizer.ASGD,
            batch_num=2,
            learning_rate=None,
        )
        self.assertRaises(
            ValueError,
            paddle.optimizer.ASGD,
            batch_num=None,
        )
        self.assertRaises(
            ValueError,
            paddle.optimizer.ASGD,
            batch_num=-2,
        )

    def test_asgd_group_dygraph(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear_1 = paddle.nn.Linear(13, 5)
        linear_2 = paddle.nn.Linear(5, 3)
        asgd = paddle.optimizer.ASGD(
            learning_rate=0.001,
            batch_num=2,
            parameters=[
                {'params': linear_1.parameters()},
                {
                    'params': linear_2.parameters(),
                    'learning_rate': 0.0001,
                },
            ],
        )
        out = linear_1(a)
        out = linear_2(out)
        out.backward()
        asgd.step()
        asgd.clear_gradients()


class TestASGDV2WeightDecay(unittest.TestCase):
    def test_weight_decay_int(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear = paddle.nn.Linear(13, 5)

        asgd = paddle.optimizer.ASGD(
            learning_rate=0.001,
            batch_num=2,
            parameters=linear.parameters(),
            weight_decay=1,
        )
        out = linear(a)
        out.backward()
        asgd.step()
        asgd.clear_gradients()


class TestASGDMultiPrecision(unittest.TestCase):
    def dygraph_asgd_mp(self, mp):
        paddle.disable_static()
        paddle.seed(10)
        paddle.set_device('gpu')
        input = paddle.randn((2, 2))
        model = paddle.nn.Linear(2, 2)
        optimizer = paddle.optimizer.ASGD(
            batch_num=2, parameters=model.parameters(), multi_precision=mp
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

    def static_asgd_mp(self, mp):
        paddle.enable_static()
        with paddle.pir_utils.OldIrGuard():
            paddle.seed(10)
            np.random.seed(10)
            exe = paddle.static.Executor('gpu')
            train_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            optimizer = paddle.optimizer.ASGD(batch_num=2, multi_precision=mp)

            if mp:
                optimizer = paddle.static.amp.decorate(
                    optimizer,
                    init_loss_scaling=128.0,
                    use_dynamic_loss_scaling=True,
                    use_pure_fp16=True,
                    use_fp16_guard=False,
                )
            with paddle.static.program_guard(train_program, startup_program):
                if mp:
                    data = paddle.static.data(
                        shape=[2, 2], name='X', dtype='float16'
                    )
                else:
                    data = paddle.static.data(
                        shape=[2, 2], name='X', dtype='float32'
                    )
                hidden = paddle.static.nn.fc(x=data, size=10)
                loss = paddle.mean(hidden)
                optimizer.minimize(loss)
            exe.run(startup_program)

            if mp:
                optimizer.amp_init(
                    place=paddle.CUDAPlace(0),
                    scope=paddle.static.global_scope(),
                )
                x = np.random.random(size=(2, 2)).astype('float16')
            else:
                x = np.random.random(size=(2, 2)).astype('float32')
            out = []
            for idx in range(5):
                (loss_data,) = exe.run(
                    train_program, feed={"X": x}, fetch_list=[loss.name]
                )
                out.append(loss_data)
            return out

    def pir_asgd_mp(self, mp):
        paddle.enable_static()
        with paddle.pir_utils.IrGuard():
            paddle.seed(10)
            np.random.seed(10)
            exe = paddle.static.Executor('gpu')
            train_program = paddle.static.Program()
            startup_program = paddle.static.Program()

            with paddle.static.program_guard(train_program, startup_program):
                model = paddle.nn.Linear(2, 10)
                optimizer = paddle.optimizer.ASGD(
                    batch_num=2,
                    multi_precision=mp,
                    parameters=model.parameters(),
                )
                if mp:
                    model, optimizer = paddle.amp.decorate(
                        models=model,
                        optimizers=optimizer,
                        level='O2',
                    )
                    scaler = paddle.amp.GradScaler(
                        init_loss_scaling=1024, use_dynamic_loss_scaling=True
                    )
                    data = paddle.static.data(
                        shape=[2, 2], name='X', dtype='float16'
                    )
                    with paddle.amp.auto_cast(
                        level='O2', dtype='float16', use_promote=True
                    ):
                        output = model(data)
                        loss = paddle.mean(output)
                    scaled = scaler.scale(loss)
                    scaler.minimize(optimizer, scaled)
                else:
                    data = paddle.static.data(
                        shape=[2, 2], name='X', dtype='float32'
                    )
                    output = model(data)
                    loss = paddle.mean(output)
                    optimizer.minimize(loss)
            exe.run(startup_program)

            if mp:
                x = np.random.random(size=(2, 2)).astype('float16')
            else:
                x = np.random.random(size=(2, 2)).astype('float32')
            out = []
            for idx in range(5):
                (loss_data,) = exe.run(
                    train_program, feed={"X": x}, fetch_list=[loss]
                )
                out.append(loss_data)
            return out

    def test_main(self):
        if not paddle.is_compiled_with_cuda():
            return
        "Test dygraph mode"
        output1_dy, params1_dy = self.dygraph_asgd_mp(mp=True)
        output2_dy, params2_dy = self.dygraph_asgd_mp(mp=False)
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
        output1_st = self.static_asgd_mp(mp=True)
        output2_st = self.static_asgd_mp(mp=False)
        for idx in range(len(output1_st)):
            np.testing.assert_allclose(
                output1_st[idx].astype('float32'),
                output2_st[idx].astype('float32'),
                rtol=1e-05,
                atol=0.1,
            )
        "Test pir graph mode"
        output1_pir = self.pir_asgd_mp(mp=True)
        output2_pir = self.pir_asgd_mp(mp=False)
        for idx in range(len(output1_st)):
            np.testing.assert_allclose(
                output1_pir[idx].astype('float32'),
                output2_pir[idx].astype('float32'),
                rtol=1e-05,
                atol=0.1,
            )


class TestASGDSimple(unittest.TestCase):
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

                optimizer = paddle.optimizer.ASGD(
                    batch_num=3,
                )

                optimizer.minimize(loss)

            exe.run(startup_program)
            out = []
            for _ in range(10):
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
            optimizer = paddle.optimizer.ASGD(
                batch_num=3, parameters=model.parameters()
            )
            for _ in range(10):
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


class TestASGDValidation:
    def setUp(self) -> None:
        self.init_all_size()
        self.init_batch_size()
        self.init_batch_num()
        self.data = np.random.random(size=(self.all_size, 2)).astype('float32')

    def init_all_size(self):
        self.all_size = 64

    def init_batch_size(self):
        self.batch_size = 8

    def init_batch_num(self):
        self.batch_num = (int)(self.all_size / self.batch_size)

    def run_validation(self) -> None:
        with dygraph_guard():
            paddle.seed(10)
            np.random.seed(10)

            param_validation = {}
            grad_validation = {}
            lr_validation = {}
            d_validation = {}
            ys_validation = {}
            y_validation = {}
            n_validation = {}

            model = paddle.nn.Linear(2, 2)
            optimizer = paddle.optimizer.ASGD(
                batch_num=self.batch_num, parameters=model.parameters()
            )

            for param in model.parameters():
                d_validation[param.name] = np.zeros(param.shape)
                ys_validation[param.name] = np.zeros(
                    [self.batch_num, *param.shape]
                )

            for i in range(5):
                data_start = i * self.batch_size % self.all_size
                data_end = data_start + self.batch_size
                cur_data = self.data[data_start:data_end]
                output = model(paddle.to_tensor(cur_data))
                loss = paddle.mean(output)
                loss = output
                loss.backward()

                for param in model.parameters():
                    param_validation[param.name] = param.numpy()

                optimizer.step()

                for param in model.parameters():
                    grad_validation[param.name] = param.grad.numpy()
                    lr_validation[param.name] = optimizer.get_lr()
                    y_validation[param.name] = ys_validation[param.name][
                        i % self.batch_num
                    ]
                    d_validation[param.name] = (
                        d_validation[param.name]
                        - y_validation[param.name]
                        + grad_validation[param.name]
                    )
                    ys_validation[param.name][i % self.batch_num] = (
                        grad_validation[param.name]
                    )
                    n_validation[param.name] = min(i + 1, self.batch_num)
                    param_validation[param.name] = (
                        param_validation[param.name]
                        - lr_validation[param.name]
                        * d_validation[param.name]
                        / n_validation[param.name]
                    )

                    np.testing.assert_allclose(
                        param.numpy(),
                        param_validation[param.name],
                    )

                optimizer.clear_grad()

    def test_main(self):
        if not paddle.is_compiled_with_cuda():
            return
        self.run_validation()


class TestASGDValidationCase1(TestASGDValidation, unittest.TestCase):
    pass


class TestASGDValidationCase2(TestASGDValidationCase1):
    def init_batch_num(self):
        self.batch_num = 2


if __name__ == "__main__":
    unittest.main()
