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
from contextlib import contextmanager

import numpy as np
from amp_base_models import AmpTestBase

import paddle
from paddle import nn
from paddle.base import core
from paddle.static import amp


@unittest.skipIf(
    not core.is_compiled_with_cuda() and not core.is_compiled_with_xpu(),
    "Require compiled with CUDA or XPU.",
)
@unittest.skipIf(
    core.is_compiled_with_cuda()
    and paddle.device.cuda.get_device_capability()[0] < 7.0,
    "run test when gpu's compute capability is at least 7.0.",
)
@unittest.skipIf(
    core.is_compiled_with_xpu()
    and core.get_xpu_device_version(0) < core.XPUVersion.XPU3,
    "run test when xpu's compute capability >= xpu3.",
)
@unittest.skipIf(
    core.is_compiled_with_xpu()
    and core.get_xpu_device_version(0) == core.XPUVersion.XPU3,
    "Bugs on XPU3, disable temporarily",
)
class TestAutoCast(AmpTestBase):
    def init_net(self):
        self._conv = paddle.nn.Conv2D(
            in_channels=1, out_channels=6, kernel_size=3, bias_attr=False
        )
        self._linear = paddle.nn.Linear(in_features=4, out_features=4)

    def test_amp_OD_level(self):
        self.init_net()
        with paddle.amp.auto_cast(level='OD'):
            out1 = self._conv(paddle.rand(shape=[1, 1, 6, 6], dtype='float32'))
            out2 = out1 + paddle.rand(shape=out1.shape, dtype='float16')
            out3 = self._linear(out2)

        self.assertEqual(out1.dtype, paddle.float16)
        self.assertEqual(out2.dtype, paddle.float32)
        self.assertEqual(out3.dtype, paddle.float32)

    def test_pir_amp_OD_level(self):
        with (
            paddle.pir_utils.IrGuard(),
            paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ),
        ):
            self.init_net()
            with paddle.amp.auto_cast(level='OD'):
                out1 = self._conv(
                    paddle.rand(shape=[1, 1, 6, 6], dtype='float32')
                )
                out2 = out1 + paddle.rand(shape=out1.shape, dtype='float16')
                out3 = self._linear(out2)

            self.assertEqual(out1.dtype, core.DataType.FLOAT16)
            self.assertEqual(out2.dtype, core.DataType.FLOAT32)
            self.assertEqual(out3.dtype, core.DataType.FLOAT32)


@unittest.skipIf(
    not core.is_compiled_with_cuda() and not core.is_compiled_with_xpu(),
    "Require compiled with CUDA or XPU.",
)
@unittest.skipIf(
    core.is_compiled_with_cuda()
    and paddle.device.cuda.get_device_capability()[0] < 7.0,
    "run test when gpu's compute capability is at least 7.0.",
)
@unittest.skipIf(
    core.is_compiled_with_xpu()
    and core.get_xpu_device_version(0) < core.XPUVersion.XPU3,
    "run test when xpu's compute capability >= xpu3.",
)
@unittest.skipIf(
    core.is_compiled_with_xpu()
    and core.get_xpu_device_version(0) == core.XPUVersion.XPU3,
    "Bugs on XPU3, disable temporarily",
)
class TestCudaAutoCast(unittest.TestCase):
    def setUp(self):
        self._conv = paddle.nn.Conv2D(1, 1, 3, bias_attr=False)
        self._linear = paddle.nn.Linear(4, 4)

    def _run_autocast_test(self, ctx):
        with ctx:
            out1 = self._conv(paddle.rand(shape=[1, 1, 6, 6], dtype='float32'))
            out2 = out1 + paddle.rand(shape=out1.shape, dtype='float16')
            out3 = self._linear(out2)

        self.assertEqual(out1.dtype, paddle.float16)
        self.assertEqual(out2.dtype, paddle.float16)
        self.assertEqual(out3.dtype, paddle.float32)

    def test_amp_autocast(self):
        self._run_autocast_test(paddle.amp.autocast(device_type='cuda'))

    def test_amp_autocast2(self):
        self._run_autocast_test(
            paddle.amp.autocast(
                device_type='cuda',
                enabled=True,
                dtype=paddle.float16,
                cache_enabled=True,
            )
        )

    def test_autocast(self):
        self._run_autocast_test(
            paddle.autocast(
                device_type='cuda',
                enabled=True,
                dtype=paddle.float16,
                cache_enabled=True,
            )
        )

    def test_cuda_amp_autocast(self):
        self._run_autocast_test(paddle.cuda.amp.autocast())

    def test_device_amp_autocast(self):
        self._run_autocast_test(paddle.device.amp.autocast())

    def test_cuda_amp_autocast_mode_autocast(self):
        self._run_autocast_test(paddle.cuda.amp.autocast_mode.autocast())


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
    not core.is_compiled_with_cuda() and not core.is_compiled_with_xpu(),
    "Require compiled with CUDA or XPU.",
)
@unittest.skipIf(
    core.is_compiled_with_cuda()
    and paddle.device.cuda.get_device_capability()[0] < 7.0,
    "run test when gpu's compute capability is at least 7.0.",
)
@unittest.skipIf(
    core.is_compiled_with_xpu()
    and core.get_xpu_device_version(0) < core.XPUVersion.XPU3,
    "run test when xpu's compute capability >= xpu3.",
)
@unittest.skipIf(
    core.is_compiled_with_xpu()
    and core.get_xpu_device_version(0) == core.XPUVersion.XPU3,
    "Bugs on XPU3, disable temporarily",
)
class TestStaticDecorate(AmpTestBase):
    def check_results(
        self, use_amp, dtype, level, use_promote, expected_op_calls
    ):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with (
            paddle.utils.unique_name.guard(),
            paddle.static.program_guard(main_program, startup_program),
        ):
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

        if paddle.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        elif paddle.device.is_compiled_with_xpu():
            place = paddle.device.XPUPlace(0)
        else:
            raise ValueError("Only support CUDA or XPU Place.")
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


@unittest.skipIf(
    not core.is_compiled_with_cuda() and not core.is_compiled_with_xpu(),
    "Require compiled with CUDA or XPU.",
)
@unittest.skipIf(
    core.is_compiled_with_cuda()
    and paddle.device.cuda.get_device_capability()[0] < 7.0,
    "run test when gpu's compute capability is at least 7.0.",
)
@unittest.skipIf(
    core.is_compiled_with_xpu()
    and core.get_xpu_device_version(0) < core.XPUVersion.XPU3,
    "run test when xpu's compute capability >= xpu3.",
)
@unittest.skipIf(
    core.is_compiled_with_xpu()
    and core.get_xpu_device_version(0) == core.XPUVersion.XPU3,
    "Bugs on XPU3, disable temporarily",
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
        op_list = paddle.base.core.get_low_precision_op_list()

        self.assertEqual(scaler._enable, False)
        self.assertEqual(scaler._use_dynamic_loss_scaling, False)
        self.assertTrue('scale' not in op_list)
        self.assertTrue('check_finite_and_unscale' not in op_list)

    def test_pir_amp_grad_scaler(self):
        with paddle.pir_utils.IrGuard():
            startup = paddle.static.Program()
            main = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                model = paddle.nn.Conv2D(3, 2, 3)
                optimizer = paddle.optimizer.SGD(
                    learning_rate=0.01, parameters=model.parameters()
                )
                model, optimizer = paddle.amp.decorate(
                    models=model,
                    optimizers=optimizer,
                )
                scaler = paddle.amp.GradScaler()
                data = paddle.static.data('data', [1, 3, 8, 8], dtype='float32')

                with paddle.amp.auto_cast(
                    custom_black_list=['conv2d'], dtype='bfloat16'
                ):
                    out = model(data)
                    loss = out.mean()
                scaled = scaler.scale(loss)
                scaler.minimize(optimizer, scaled)

                if paddle.is_compiled_with_cuda():
                    place = paddle.CUDAPlace(0)
                elif paddle.device.is_compiled_with_xpu():
                    place = paddle.device.XPUPlace(0)
                else:
                    raise ValueError("Only support CUDA or XPU Place.")
                exe = paddle.static.Executor(place)
                exe.run(startup)
                paddle.amp.debugging.enable_operator_stats_collection()
                exe.run(
                    main,
                    feed={'data': np.random.rand(1, 3, 8, 8).astype('float32')},
                    fetch_list=[loss],
                )
                paddle.amp.debugging.disable_operator_stats_collection()
                op_list = paddle.base.core.get_low_precision_op_list()

                self.assertEqual(scaler._enable, False)
                self.assertEqual(scaler._use_dynamic_loss_scaling, False)
                self.assertTrue('pd_op.scale' not in op_list)
                self.assertTrue(
                    'pd_op.check_finite_and_unscale_' not in op_list
                )


@unittest.skipIf(
    not core.is_compiled_with_cuda() and not core.is_compiled_with_xpu(),
    "Require compiled with CUDA or XPU.",
)
@unittest.skipIf(
    core.is_compiled_with_cuda()
    and paddle.device.cuda.get_device_capability()[0] < 7.0,
    "run test when gpu's compute capability is at least 7.0.",
)
@unittest.skipIf(
    core.is_compiled_with_xpu()
    and core.get_xpu_device_version(0) < core.XPUVersion.XPU3,
    "run test when xpu's compute capability >= xpu3.",
)
@unittest.skipIf(
    core.is_compiled_with_xpu()
    and core.get_xpu_device_version(0) == core.XPUVersion.XPU3,
    "Bugs on XPU3, disable temporarily",
)
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


# Copy from ../dygraph_to_static/dygraph_to_static_utils.py
@contextmanager
def pir_dygraph_guard():
    in_dygraph_mode = paddle.in_dynamic_mode()
    with paddle.pir_utils.IrGuard():
        if in_dygraph_mode:
            paddle.disable_static()
        yield


@unittest.skipIf(
    not core.is_compiled_with_cuda() and not core.is_compiled_with_xpu(),
    "Require compiled with CUDA or XPU.",
)
@unittest.skipIf(
    core.is_compiled_with_cuda()
    and paddle.device.cuda.get_device_capability()[0] < 7.0,
    "run test when gpu's compute capability is at least 7.0.",
)
@unittest.skipIf(
    core.is_compiled_with_xpu()
    and core.get_xpu_device_version(0) < core.XPUVersion.XPU3,
    "run test when xpu's compute capability >= xpu3.",
)
@unittest.skipIf(
    core.is_compiled_with_xpu()
    and core.get_xpu_device_version(0) == core.XPUVersion.XPU3,
    "Bugs on XPU3, disable temporarily",
)
class TestDy2STWithSetValue(AmpTestBase):
    def test_op_called_as_expected(self):
        if paddle.framework.use_pir_api():
            return
        expected_fp16_calls = {
            "cast": 1,
            "layer_norm": 1,
            "scale": 3,
            "set_value": 1,
        }

        func = SimpleModelIncludeSetValue()
        func = paddle.amp.decorate(func, level='O2')
        func = paddle.jit.to_static(func, full_graph=True, backend=None)
        input = paddle.randn((2, 3))

        with paddle.amp.auto_cast(level='O2', use_promote=False):
            res = func(input)
            loss = res.sum()
            prog = func.forward.get_concrete_program(input)[1].forward_program
            amp.debugging.collect_operator_stats(prog)
            op_stats_list = amp.debugging._get_op_stats_list(prog)
        loss.backward()
        self._check_op_calls(
            op_stats_list[0], expected_fp16_calls=expected_fp16_calls
        )

    def test_pir_op_called_as_expected(self):
        expected_fp16_calls = {
            "pd_op.layer_norm": 1,
            "pd_op.scale": 1,
            "pd_op.scale_": 2,
            "pd_op.set_value_with_tensor_": 1,
        }

        with pir_dygraph_guard():
            func = SimpleModelIncludeSetValue()
            func = paddle.amp.decorate(func, level='O2')
            func = paddle.jit.to_static(func, full_graph=True, backend=None)
            input = paddle.randn((2, 3))

            paddle.amp.debugging.enable_operator_stats_collection()
            with paddle.amp.auto_cast(level='O2', use_promote=False):
                res = func(input)
                loss = res.sum()
                paddle.amp.debugging.disable_operator_stats_collection()
                op_stats = paddle.base.core.get_low_precision_op_list()

            loss.backward()
            self._check_op_calls(
                op_stats, expected_fp16_calls=expected_fp16_calls
            )


if __name__ == '__main__':
    unittest.main()
