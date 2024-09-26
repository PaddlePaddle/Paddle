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

import paddle
from paddle import base
from paddle.base import core, in_pir_mode


def create_selected_rows_and_tensor(
    scope, place, height, row_num, embedding_size
):
    sr = scope.var("@selected_rows@").get_selected_rows()
    tensor = scope.var("grad").get_tensor()

    rows = np.random.random_integers(
        low=0,
        high=height - 1,
        size=[
            row_num,
        ],
    ).astype('int64')
    sr_val = np.random.random(size=[row_num, embedding_size]).astype('float32')

    sr.set_height(height)
    sr.set_rows(rows)
    sr.get_tensor().set(sr_val, place)

    tensor_val = np.zeros(shape=[height, embedding_size], dtype='float32')
    for i in range(row_num):
        row = rows[i]
        tensor_val[row, :] = tensor_val[row, :] + sr_val[i, :]

    tensor.set(tensor_val, place)
    return tensor_val, sr_val


class TestBase(unittest.TestCase):
    def setup(
        self, place, is_sparse, centered, size, row_num=None, epsilon=1e-6
    ):
        np.random.seed(5)  # fix seed

        self.scope = base.global_scope()
        self.place = place

        self.param_name = "param"
        self.param = np.random.random(size).astype("float32")

        self.mean_square_name = "mean_square"
        self.mean_square = np.random.uniform(low=1, high=2, size=size).astype(
            "float32"
        )

        self.mean_grad_name = "mean_grad"
        self.mean_grad = np.random.random(size).astype("float32")

        self.lr_name = "lr"
        self.learning_rate = np.array([0.01]).astype("float32")

        self.grad_name = "grad"

        self.is_sparse = is_sparse
        if self.is_sparse:
            self.grad_sr_name = "@selected_rows@"
            self.grad, self.grad_sr = create_selected_rows_and_tensor(
                self.scope, place, size[0], row_num, size[1]
            )
        else:
            self.grad = np.random.random(size).astype("float32")
            grad_tensor = self.scope.var(self.grad_name).get_tensor()
            grad_tensor.set(self.grad, place)

        self.moment_name = "moment"
        self.moment = np.random.uniform(low=0, high=1, size=size).astype(
            "float32"
        )

        self.epsilon = epsilon
        self.decay = 0.9
        self.momentum = 0.1
        self.centered = centered

        self.ms_out = (
            self.decay * self.mean_square
            + (1 - self.decay) * self.grad * self.grad
        )
        if centered:
            self.mg_out = (
                self.decay * self.mean_grad + (1 - self.decay) * self.grad
            )
            self.moment_out = (
                self.momentum * self.moment
                + self.learning_rate
                * self.grad
                / np.sqrt(self.ms_out - np.square(self.mg_out) + self.epsilon)
            )
        else:
            self.moment_out = (
                self.momentum * self.moment
                + self.learning_rate
                * self.grad
                / np.sqrt(self.ms_out + self.epsilon)
            )

        self.param_out = self.param - self.moment_out

        # create and initialize Param Variable
        self.param_tensor = self.scope.var(self.param_name).get_tensor()
        self.param_tensor.set(self.param, place)

        self.mean_square_tensor = self.scope.var(
            self.mean_square_name
        ).get_tensor()
        self.mean_square_tensor.set(self.mean_square, place)

        lr = self.scope.var(self.lr_name).get_tensor()
        lr.set(self.learning_rate, place)

        self.moment_tensor = self.scope.var(self.moment_name).get_tensor()
        self.moment_tensor.set(self.moment, place)

        if self.centered:
            self.mean_grad_tensor = self.scope.var(
                self.mean_grad_name
            ).get_tensor()
            self.mean_grad_tensor.set(self.mean_grad, place)

    def check(self, actual_t, expect_t, place, out_name, atol=1e-5):
        np.testing.assert_allclose(
            actual_t,
            expect_t,
            rtol=1e-05,
            atol=atol,
            err_msg='Output ('
            + out_name
            + ') has diff at '
            + str(place)
            + '\nExpect '
            + str(expect_t)
            + '\n'
            + 'But Got'
            + str(actual_t),
        )


class TestRmspropOp(TestBase):
    def check_with_place(
        self, place, is_sparse, centered, size, row_num=None, epsilon=1e-6
    ):
        self.setup(place, is_sparse, centered, size, row_num, epsilon)
        self.run_and_check()

    def run_and_check(self):
        grad_name = self.grad_sr_name if self.is_sparse else self.grad_name

        kwargs = {
            'Param': self.param_name,
            'Grad': grad_name,
            'MeanSquare': self.mean_square_name,
            'Moment': self.moment_name,
            'LearningRate': self.lr_name,
            'ParamOut': self.param_name,
            'MeanSquareOut': self.mean_square_name,
            'MomentOut': self.moment_name,
            'epsilon': self.epsilon,
            'decay': self.decay,
            'momentum': self.momentum,
            'centered': self.centered,
        }

        if self.centered:
            kwargs['MeanGrad'] = self.mean_grad_name
            kwargs['MeanGradOut'] = self.mean_grad_name

        rmsprop_op = Operator('rmsprop', **kwargs)
        atol = 1e-6

        rmsprop_op.run(self.scope, self.place)

        self.check(
            np.array(self.mean_square_tensor),
            self.ms_out,
            self.place,
            self.mean_square_name,
            atol=atol,
        )
        self.check(
            np.array(self.moment_tensor),
            self.moment_out,
            self.place,
            self.moment_name,
            atol=atol,
        )
        self.check(
            np.array(self.param_tensor),
            self.param_out,
            self.place,
            self.param_name,
            atol=atol,
        )

        if self.centered:
            self.check(
                np.array(self.mean_grad_tensor),
                self.mg_out,
                self.place,
                self.mean_grad_name,
            )

    def test_rmsprop(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(core.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))

        size = (128, 320)
        for place in places:
            for centered in [False, True]:
                with base.scope_guard(core.Scope()):
                    self.check_with_place(
                        place, is_sparse=False, centered=centered, size=size
                    )

                with base.scope_guard(core.Scope()):
                    self.check_with_place(
                        place,
                        is_sparse=True,
                        centered=centered,
                        row_num=512,
                        size=size,
                    )

                with base.scope_guard(core.Scope()):
                    self.check_with_place(
                        place,
                        is_sparse=True,
                        centered=centered,
                        row_num=60,
                        size=size,
                    )


class TestRMSPropV2(unittest.TestCase):
    def test_rmsprop_dygraph(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear = paddle.nn.Linear(13, 5)
        # This can be any optimizer supported by dygraph.
        adam = paddle.optimizer.RMSProp(
            learning_rate=0.01,
            parameters=linear.parameters(),
            weight_decay=0.01,
        )
        out = linear(a)
        out.backward()
        adam.step()
        adam.clear_gradients()

    def test_rmsprop(self):
        paddle.enable_static()
        place = base.CPUPlace()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name='x', shape=[-1, 13], dtype='float32')
            y = paddle.static.data(name='y', shape=[-1, 1], dtype='float32')
            y_predict = paddle.nn.Linear(
                in_features=x.shape[-1], out_features=1
            )(x)
            cost = paddle.nn.functional.square_error_cost(
                input=y_predict, label=y
            )
            avg_cost = paddle.mean(cost)

            rms_optimizer = paddle.optimizer.RMSProp(learning_rate=0.1)
            rms_optimizer.minimize(avg_cost)

            fetch_list = [avg_cost]
            train_reader = paddle.batch(
                paddle.dataset.uci_housing.train(), batch_size=1
            )
            feeder = base.DataFeeder(place=place, feed_list=[x, y])
            exe = base.Executor(place)
            exe.run(startup)
            for data in train_reader():
                exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)

    def test_raise_error(self):
        self.assertRaises(ValueError, paddle.optimizer.RMSProp, None)
        self.assertRaises(
            ValueError, paddle.optimizer.RMSProp, learning_rate=0.1, rho=None
        )
        self.assertRaises(
            ValueError,
            paddle.optimizer.RMSProp,
            learning_rate=0.1,
            epsilon=None,
        )
        self.assertRaises(
            ValueError,
            paddle.optimizer.RMSProp,
            learning_rate=0.1,
            momentum=None,
        )

    def test_rmsprop_op_invalid_input(self):
        paddle.disable_static()
        linear = paddle.nn.Linear(10, 10)
        with self.assertRaises(ValueError):
            adam = paddle.optimizer.RMSProp(
                0.1, epsilon=-1, parameters=linear.parameters()
            )
        with self.assertRaises(ValueError):
            adam = paddle.optimizer.RMSProp(
                0.1, momentum=-1, parameters=linear.parameters()
            )
        with self.assertRaises(ValueError):
            adam = paddle.optimizer.RMSProp(
                0.1, rho=-1, parameters=linear.parameters()
            )


class TestRMSPropV2WeightDecay(unittest.TestCase):
    def test_weight_decay_int(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear = paddle.nn.Linear(13, 5)
        # This can be any optimizer supported by dygraph.
        adam = paddle.optimizer.RMSProp(
            learning_rate=0.01,
            parameters=linear.parameters(),
            weight_decay=1,
        )
        out = linear(a)
        out.backward()
        adam.step()
        adam.clear_gradients()


class TestRMSPropV2Group(TestRMSPropV2):
    def test_rmsprop_dygraph(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear_1 = paddle.nn.Linear(13, 5)
        linear_2 = paddle.nn.Linear(5, 3)
        # This can be any optimizer supported by dygraph.
        adam = paddle.optimizer.RMSProp(
            learning_rate=0.01,
            parameters=[
                {'params': linear_1.parameters()},
                {'params': linear_2.parameters(), 'weight_decay': 0.001},
            ],
            weight_decay=0.01,
        )
        out = linear_1(a)
        out = linear_2(out)
        out.backward()
        adam.step()
        adam.clear_gradients()


class TestRMSOpMultiPrecision(unittest.TestCase):
    def _test_rms_op_dygraph_place_amp(self, place, use_amp=False):
        import paddle

        paddle.disable_static()
        paddle.seed(10)
        paddle.set_device(place)

        input = paddle.randn((5, 5))

        model = paddle.nn.Linear(5, 5)

        optimizer = paddle.optimizer.RMSProp(
            learning_rate=0.01,
            parameters=model.parameters(),
            weight_decay=0.01,
        )
        optimizer._multi_precision = use_amp
        for idx in range(2):
            if place == 'gpu' and use_amp:
                model = paddle.amp.decorate(models=model, level='O2')
                scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

            if place == 'gpu' and use_amp:
                with paddle.amp.auto_cast(level='O2'):
                    output = model(input)
                    loss = paddle.mean(output)
                scaled = scaler.scale(loss)
                scaled.backward()
                scaler.step(optimizer)
                optimizer.clear_grad()
            else:
                output = model(input)
                loss = paddle.mean(output)
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()
        paddle.enable_static()

    def _get_places(self):
        import paddle

        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.is_compiled_with_cuda()
        ):
            places.append('cpu')
        if paddle.is_compiled_with_cuda():
            places.append('gpu')
        return places

    def test_main(self):
        for place in self._get_places():
            use_amp_list = [True, False]
            for use_amp in use_amp_list:
                self._test_rms_op_dygraph_place_amp(place, use_amp)


class TestRMSPropMultiPrecision2_0(unittest.TestCase):
    def dygraph_rmsprop_mp(self, mp, use_amp):
        paddle.disable_static()
        paddle.seed(100)
        paddle.set_device('gpu')
        input = paddle.randn((2, 2))
        model = paddle.nn.Linear(2, 2)
        optimizer = paddle.optimizer.RMSProp(0.5, parameters=model.parameters())
        optimizer._multi_precision = mp
        if use_amp:
            model = paddle.amp.decorate(models=model, level='O2')
            scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

        for idx in range(5):
            if use_amp:
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
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()

        return output, model.parameters()

    def static_rmsprop_mp(self, mp, use_amp):
        paddle.enable_static()
        paddle.seed(100)
        np.random.seed(100)
        exe = paddle.static.Executor('gpu')
        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()

        with paddle.static.program_guard(train_program, startup_program):
            if in_pir_mode():
                optimizer = paddle.optimizer.RMSProp(0.1)
                optimizer._multi_precision = mp
                linear = paddle.nn.Linear(2, 2)

                if mp:
                    linear, optimizer = paddle.amp.decorate(
                        models=linear,
                        optimizers=optimizer,
                        level='O2',
                        dtype='float16',
                    )
            else:
                optimizer = paddle.optimizer.RMSProp(0.1)
                optimizer._multi_precision = mp
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

    def pir_rmsprop_mp(self, mp, use_amp):
        with paddle.pir_utils.IrGuard():
            paddle.seed(100)
            np.random.seed(100)
            exe = paddle.static.Executor('gpu')
            train_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            optimizer = paddle.optimizer.RMSProp(0.1)
            optimizer._multi_precision = mp

            if use_amp:
                optimizer = paddle.static.amp.decorate(
                    optimizer,
                    init_loss_scaling=128.0,
                    use_dynamic_loss_scaling=True,
                    use_pure_fp16=True,
                    use_fp16_guard=False,
                )
            with paddle.static.program_guard(train_program, startup_program):
                if use_amp:
                    data = paddle.static.data(
                        shape=[2, 2], name='X', dtype='float16'
                    )
                else:
                    data = paddle.static.data(
                        shape=[2, 2], name='X', dtype='float32'
                    )
                hidden = paddle.nn.Linear(
                    in_features=data.shape[-1], out_features=10
                )(data)
                loss = paddle.mean(hidden)
                optimizer.minimize(loss)
            exe.run(startup_program)

            if use_amp:
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
                    train_program, feed={"X": x}, fetch_list=[loss]
                )
                out.append(loss_data)
            return out

    def test_main(self):
        if not paddle.is_compiled_with_cuda():
            return
        "Test dygraph mode"
        output1_dy, params1_dy = self.dygraph_rmsprop_mp(use_amp=True, mp=True)
        output2_dy, params2_dy = self.dygraph_rmsprop_mp(
            use_amp=False, mp=False
        )
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
        "Test static mode"
        output1_st = self.static_rmsprop_mp(use_amp=True, mp=True)
        output2_st = self.static_rmsprop_mp(use_amp=False, mp=False)
        for idx in range(len(output1_st)):
            np.testing.assert_allclose(
                output1_st[idx].astype('float32'),
                output2_st[idx].astype('float32'),
                rtol=1e-05,
                atol=0.1,
            )
        # NOT support amp training "Test pir mode"
        # output1_pir = self.pir_rmsprop_mp(use_amp=True, mp=True)
        # output2_pir = self.pir_rmsprop_mp(use_amp=False, mp=False)
        # for idx in range(len(output1_pir)):
        #     np.testing.assert_allclose(
        #         output1_pir[idx].astype('float32'),
        #         output2_pir[idx].astype('float32'),
        #         rtol=1e-05,
        #         atol=0.1,
        #     )


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
