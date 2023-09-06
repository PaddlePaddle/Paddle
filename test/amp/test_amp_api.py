# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from amp_base_models import AmpTestBase

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.fluid import core
from paddle.static import amp


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or paddle.device.cuda.get_device_capability()[0] < 7.0,
    "run test when gpu's compute capability is at least 7.0.",
)
class TestAutoCast(AmpTestBase):
    def setUp(self):
        self._conv = paddle.nn.Conv2D(
            in_channels=1, out_channels=6, kernel_size=3, bias_attr=False
        )
        self._linear = paddle.nn.Linear(in_features=4, out_features=4)

    def test_amp_OD_level(self):
        with paddle.amp.auto_cast(level='OD'):
            out1 = self._conv(paddle.rand(shape=[1, 1, 6, 6], dtype='float32'))
            out2 = out1 + paddle.rand(shape=out1.shape, dtype='float16')
            out3 = self._linear(out2)

        self.assertEqual(out1.dtype, paddle.float16)
        self.assertEqual(out2.dtype, paddle.float32)
        self.assertEqual(out3.dtype, paddle.float32)


class SimpleConvNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self._conv = paddle.nn.Conv2D(
            in_channels=1, out_channels=6, kernel_size=3, bias_attr=False
        )
        self._linear = paddle.nn.Linear(in_features=4, out_features=4)

    def forward(self, x):
        out1 = self._conv(paddle.rand(shape=[1, 1, 6, 6], dtype='float32'))
        out2 = out1 + paddle.rand(shape=out1.shape, dtype='float16')
        out3 = self._linear(out2)
        return out3


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or paddle.device.cuda.get_device_capability()[0] < 7.0,
    "run test when gpu's compute capability is at least 7.0.",
)
class TestStaticDecorate(AmpTestBase):
    def check_results(
        self, use_amp, dtype, level, use_promote, expected_op_calls
    ):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.utils.unique_name.guard():
            with paddle.static.program_guard(main_program, startup_program):
                model = SimpleConvNet()
                x = paddle.static.data(
                    name='input', shape=[None, 1, 6, 6], dtype='float32'
                )
                out = model(x)
                loss = paddle.mean(out)
                optimizer = paddle.optimizer.Adadelta(learning_rate=0.001)
                optimizer = paddle.static.amp.decorate(
                    optimizer,
                    init_loss_scaling=128.0,
                    use_dynamic_loss_scaling=True,
                    level=level,
                )
                optimizer.minimize(loss)

        feed_vars = [x]
        fetch_vars = [loss]
        self.assertEqual(main_program.num_blocks, 1)

        amp.debugging.collect_operator_stats(main_program)
        op_stats_list = amp.debugging._get_op_stats_list(main_program)

        self._check_op_calls(
            op_stats_list[0], expected_fp16_calls=expected_op_calls
        )

        place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)

        max_iters = 2
        x_fp32 = np.random.random(size=[1, 1, 6, 6]).astype("float32")
        losses_o1 = self.run_program(
            main_program,
            startup_program,
            optimizer,
            feed_vars,
            fetch_vars,
            place,
            exe,
            x_fp32,
            max_iters,
            dtype,
            level,
        )

    def test_static_amp_OD(self):
        paddle.enable_static()
        expected_fp16_calls = {
            "conv2d": 1,
            "elementwise_add": 0,
            "matmul_v2": 1,
            "reduce_mean": 0,
        }
        self.check_results(
            True,
            'float16',
            'OD',
            use_promote=True,
            expected_op_calls=expected_fp16_calls,
        )
        paddle.disable_static()


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or paddle.device.cuda.get_device_capability()[0] < 7.0,
    "run test when gpu's compute capability is at least 7.0.",
)
class TestGradScaler(AmpTestBase):
    def test_amp_grad_scaler(self):
        model = paddle.nn.Conv2D(3, 2, 3)
        optimizer = paddle.optimizer.SGD(
            learning_rate=0.01, parameters=model.parameters()
        )
        scaler = paddle.amp.GradScaler()
        data = paddle.rand([1, 3, 8, 8], dtype='float32')
        paddle.amp.debugging.enable_operator_stats_collection()
        with paddle.amp.auto_cast(
            custom_black_list=['conv2d'], dtype='bfloat16'
        ):
            out = model(data)
            loss = out.mean()
        scaled = scaler.scale(loss)
        scaled.backward()
        scaler.minimize(optimizer, scaled)
        optimizer.clear_grad()
        paddle.amp.debugging.disable_operator_stats_collection()
        op_list = paddle.fluid.core.get_low_precision_op_list()

        self.assertEqual(scaler._enable, False)
        self.assertEqual(scaler._use_dynamic_loss_scaling, False)
        self.assertTrue('scale' not in op_list)
        self.assertTrue('check_finite_and_unscale' not in op_list)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or paddle.device.cuda.get_device_capability()[0] < 7.0,
    "run test when gpu's compute capability is at least 7.0.",
)
class TestFp16Guard(AmpTestBase):
    def test_fp16_gurad(self):
        paddle.enable_static()

        def run_example_code():
            place = paddle.CUDAPlace(0)
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()

            exe = paddle.static.Executor(place)

            fetch_vars = []
            # 1) Use fp16_guard to control the range of fp16 kernels used.
            with paddle.static.program_guard(main_program, startup_program):
                with paddle.static.amp.fp16_guard():
                    data = paddle.static.data(
                        name='X', shape=[None, 1, 28, 28], dtype='float32'
                    )
                    conv2d = paddle.static.nn.conv2d(
                        input=data, num_filters=6, filter_size=3
                    )
                    bn = paddle.static.nn.batch_norm(input=conv2d, act="relu")

                pool = F.max_pool2d(bn, kernel_size=2, stride=2)
                hidden = paddle.static.nn.fc(pool, size=10)
                loss = paddle.mean(hidden)
                fetch_vars = [loss]
                # 2) Create the optimizer and set `multi_precision` to True.
                # Setting `multi_precision` to True can avoid the poor accuracy
                # or the slow convergence in a way.
                optimizer = paddle.optimizer.Momentum(
                    learning_rate=0.01, multi_precision=True
                )
                # 3) These ops in `custom_black_list` will keep in the float32 computation type.
                amp_list = paddle.static.amp.CustomOpLists(
                    custom_black_list=['pool2d']
                )
                # 4) The entry of Paddle AMP.
                # Enable pure fp16 training by setting `use_pure_fp16` to True.
                optimizer = paddle.static.amp.decorate(
                    optimizer,
                    amp_list,
                    init_loss_scaling=128.0,
                    use_dynamic_loss_scaling=True,
                    use_pure_fp16=True,
                )
                # If you don't use the default_startup_program(), you sholud pass
                # your defined `startup_program` into `minimize`.
                optimizer.minimize(loss)

            exe.run(startup_program)
            # 5) Use `amp_init` after FP32 parameters initialization(such as `exe.run(startup_program)`).
            # If you want to perform the testing process, you should pass `test_program` into `amp_init`.
            optimizer.amp_init(place, scope=paddle.static.global_scope())

            x_fp32 = np.random.random(size=[1, 1, 28, 28]).astype("float32")
            (loss_data,) = exe.run(
                main_program, feed={"X": x_fp32}, fetch_list=[loss.name]
            )

            self.assertEqual(
                paddle.static.global_scope()
                .find_var("conv2d_0.b_0")
                .get_tensor()
                ._dtype(),
                paddle.float16,
            )
            self.assertEqual(
                paddle.static.global_scope()
                .find_var("fc_0.b_0")
                .get_tensor()
                ._dtype(),
                paddle.float32,
            )

        if (
            paddle.is_compiled_with_cuda()
            and len(paddle.static.cuda_places()) > 0
        ):
            run_example_code()
        paddle.disable_static()


class SimpleModelIncludeSetValue(nn.Layer):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(3)

    def forward(self, x):
        x = x + 1
        tmp = x * 1
        y = self.norm(tmp)
        x[:] = y

        z = x * 1
        return z


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or paddle.device.cuda.get_device_capability()[0] < 7.0,
    "run test when gpu's compute capability is at least 7.0.",
)
class TestDy2STWithSetValue(AmpTestBase):
    def test_op_called_as_expected(self):
        expected_fp16_calls = {
            "cast": 0,
            "layer_norm": 1,
            "scale": 3,
            "set_value": 1,
        }

        func = SimpleModelIncludeSetValue()
        func = paddle.amp.decorate(func, level='O2')
        func = paddle.jit.to_static(func)
        input = paddle.randn((2, 3))

        with paddle.amp.auto_cast(level='O2'):
            res = func(input)
            loss = res.sum()
            prog = func.forward.get_concrete_program(input)[1].forward_program
            amp.debugging.collect_operator_stats(prog)
            op_stats_list = amp.debugging._get_op_stats_list(prog)
        loss.backward()
        self._check_op_calls(
            op_stats_list[0], expected_fp16_calls=expected_fp16_calls
        )


if __name__ == '__main__':
    unittest.main()
