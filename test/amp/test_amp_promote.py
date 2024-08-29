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
from amp_base_models import AmpTestBase, build_conv_model

import paddle
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
class TestStaticAmpPromoteStats(AmpTestBase):
    def check_promote_results(
        self, use_amp, dtype, level, use_promote, expected_op_calls, debug_info
    ):
        paddle.enable_static()
        (
            main_program,
            startup_program,
            optimizer,
            feed_vars,
            fetch_vars,
        ) = build_conv_model(use_amp, dtype, level, use_promote)
        self.assertEqual(main_program.num_blocks, 1)

        amp.debugging.collect_operator_stats(main_program)
        op_stats_list = amp.debugging._get_op_stats_list(main_program)

        self._check_op_calls(
            op_stats_list[0],
            expected_fp16_calls=expected_op_calls,
            debug_info=debug_info,
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
        paddle.disable_static()

    def test_static_amp_o1(self):
        expected_fp16_calls = {
            "conv2d": 1,
            "elementwise_add": 0,
            "relu": 0,
            "matmul_v2": 1,
            "softmax": 0,
            "reduce_mean": 0,
            "adamw": 0,
        }
        with paddle.pir_utils.OldIrGuard():
            self.check_promote_results(
                True,
                'float16',
                'O1',
                use_promote=True,
                expected_op_calls=expected_fp16_calls,
                debug_info="TestStaticAmpPromoteStats/test_static_amp_o1",
            )

    def test_static_amp_o2(self):
        expected_fp16_calls = {
            "conv2d": 1,
            "elementwise_add": 2,
            "relu": 0,
            "matmul_v2": 1,
            "softmax": 1,
            "reduce_mean": 1,
            "adamw": 4,
        }
        with paddle.pir_utils.OldIrGuard():
            self.check_promote_results(
                True,
                'float16',
                'O2',
                use_promote=True,
                expected_op_calls=expected_fp16_calls,
                debug_info="TestStaticAmpPromoteStats/test_static_amp_o2",
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
class TestEagerAmpPromoteStats(AmpTestBase):
    def check_promote_results(
        self, dtype, level, use_promote, expected_op_calls, debug_info
    ):
        model, optimizer, scaler = build_conv_model(
            use_amp=True,
            amp_dtype=dtype,
            amp_level=level,
            use_promote=use_promote,
        )
        model.train()

        paddle.amp.debugging.enable_operator_stats_collection()
        with paddle.amp.auto_cast(
            enable=True, dtype=dtype, level=level, use_promote=use_promote
        ):
            x = paddle.rand(shape=[1, 1, 6, 6], dtype='float32')
            out = model(x)
            loss = paddle.mean(out)
        scaled = scaler.scale(loss)
        scaled.backward()
        scaler.minimize(optimizer, scaled)
        optimizer.clear_grad()
        paddle.amp.debugging.disable_operator_stats_collection()
        op_stats = paddle.base.core.get_low_precision_op_list()

        self._check_op_calls(
            op_stats,
            expected_fp16_calls=expected_op_calls,
            debug_info=debug_info,
        )

    def test_o2_promote_on(self):
        expected_fp16_calls = {
            "conv2d": 1,
            "elementwise_add": 2,
            "relu": 0,
            "matmul_v2": 1,
            "softmax": 1,
            "reduce_mean": 1,
            "adamw_": 4,
        }
        self.check_promote_results(
            'float16',
            'O2',
            use_promote=True,
            expected_op_calls=expected_fp16_calls,
            debug_info="TestEagerAmpPromoteStats/test_o2_promote_on",
        )

    def test_o2_promote_off(self):
        expected_fp16_calls = {
            "conv2d": 1,
            "elementwise_add": 2,
            "relu": 1,
            "matmul_v2": 1,
            "softmax": 1,
            "reduce_mean": 1,
            "adamw_": 4,
        }
        self.check_promote_results(
            'float16',
            'O2',
            use_promote=False,
            expected_op_calls=expected_fp16_calls,
            debug_info="TestEagerAmpPromoteStats/test_o2_promote_off",
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
class TestPirAmpPromoteStats(AmpTestBase):
    def check_promote_results(
        self, dtype, level, use_promote, expected_op_calls, debug_info
    ):
        with paddle.pir_utils.IrGuard():
            startup = paddle.static.Program()
            main = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                model, optimizer, scaler = build_conv_model(
                    use_amp=True,
                    amp_dtype=dtype,
                    amp_level=level,
                    use_promote=use_promote,
                )
                model.train()

                with paddle.amp.auto_cast(
                    enable=True,
                    dtype=dtype,
                    level=level,
                    use_promote=use_promote,
                ):
                    x = paddle.static.data(
                        'x', shape=[1, 1, 6, 6], dtype='float32'
                    )
                    out = model(x)
                    loss = paddle.mean(out)
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
                    feed={
                        'x': np.random.random([1, 1, 6, 6]).astype('float32'),
                    },
                    fetch_list=[loss],
                )
                paddle.amp.debugging.disable_operator_stats_collection()
                op_stats = paddle.base.core.get_low_precision_op_list()

                self._check_op_calls(
                    op_stats,
                    expected_fp16_calls=expected_op_calls,
                    debug_info=debug_info,
                )

    def test_o2_promote_on(self):
        paddle.set_flags({"FLAGS_pir_apply_inplace_pass": 0})
        expected_fp16_calls = {
            "pd_op.conv2d": 1,
            "pd_op.add": 2,
            "pd_op.relu": 0,
            "pd_op.matmul": 1,
            "pd_op.softmax": 1,
            "pd_op.mean": 1,
            "pd_op.adamw_": 4,
        }
        self.check_promote_results(
            'float16',
            'O2',
            use_promote=True,
            expected_op_calls=expected_fp16_calls,
            debug_info="TestEagerAmpPromoteStats/test_o2_promote_on",
        )

    def test_o2_promote_off(self):
        paddle.set_flags({"FLAGS_pir_apply_inplace_pass": 0})
        expected_fp16_calls = {
            "pd_op.conv2d": 1,
            "pd_op.add": 2,
            "pd_op.relu": 1,
            "pd_op.matmul": 1,
            "pd_op.softmax": 1,
            "pd_op.mean": 1,
            "pd_op.adamw_": 4,
        }
        self.check_promote_results(
            'float16',
            'O2',
            use_promote=False,
            expected_op_calls=expected_fp16_calls,
            debug_info="TestEagerAmpPromoteStats/test_o2_promote_off",
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda() and not core.is_compiled_with_xpu(),
    "Require compiled with CUDA or XPU.",
)
@unittest.skipIf(
    core.is_compiled_with_cuda()
    and not paddle.device.cuda.get_device_capability()[0] < 7.0,
    "run test when gpu's compute capability is at least 7.0.",
)
@unittest.skipIf(
    core.is_compiled_with_xpu()
    and core.get_xpu_device_version(0) < core.XPUVersion.XPU3,
    "run test when xpu's compute capability >= xpu3.",
)
class TestEagerAmpPromoteSimple(AmpTestBase):
    def setUp(self):
        self._conv = paddle.nn.Conv2D(
            in_channels=1, out_channels=6, kernel_size=3, bias_attr=False
        )
        self._linear = paddle.nn.Linear(in_features=4, out_features=4)

    def test_o2_use_promote_on(self):
        with paddle.amp.auto_cast(level='O2'):
            x = paddle.rand(shape=[1, 1, 6, 6], dtype='float32')
            conv_out = self._conv(x)
            y = paddle.rand(shape=conv_out.shape, dtype='float16')
            add_out = conv_out + y
            linear_out = self._linear(add_out)

        self.assertEqual(conv_out.dtype, paddle.float16)
        self.assertEqual(add_out.dtype, paddle.float16)
        self.assertEqual(linear_out.dtype, paddle.float32)

    def test_o2_use_promote_off(self):
        with paddle.amp.auto_cast(level='O2', use_promote=False):
            x = paddle.rand(shape=[1, 1, 6, 6], dtype='float32')
            conv_out = self._conv(x)
            y = paddle.rand(shape=conv_out.shape, dtype='float16')
            add_out = conv_out + y
            linear_out = self._linear(add_out)

        self.assertEqual(conv_out.dtype, paddle.float16)
        self.assertEqual(add_out.dtype, paddle.float16)
        self.assertEqual(linear_out.dtype, paddle.float16)


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
class TestPirAmpPromoteSimple(AmpTestBase):
    def init_net(self):
        self._conv = paddle.nn.Conv2D(
            in_channels=1, out_channels=6, kernel_size=3, bias_attr=False
        )
        self._linear = paddle.nn.Linear(in_features=4, out_features=4)

    def test_o2_use_promote_on(self):
        with paddle.pir_utils.IrGuard():
            startup = paddle.static.Program()
            main = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                self.init_net()
                with paddle.amp.auto_cast(level='O2'):
                    x = paddle.rand(shape=[1, 1, 6, 6], dtype='float32')
                    conv_out = self._conv(x)
                    y = paddle.rand(shape=conv_out.shape, dtype='float16')
                    add_out = conv_out + y
                    linear_out = self._linear(add_out)

            self.assertEqual(conv_out.dtype, paddle.float16)
            self.assertEqual(add_out.dtype, paddle.float16)
            self.assertEqual(linear_out.dtype, paddle.float32)

    def test_o2_use_promote_off(self):
        with paddle.pir_utils.IrGuard():
            startup = paddle.static.Program()
            main = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                self.init_net()
                with paddle.amp.auto_cast(level='O2', use_promote=False):
                    x = paddle.rand(shape=[1, 1, 6, 6], dtype='float32')
                    conv_out = self._conv(x)
                    y = paddle.rand(shape=conv_out.shape, dtype='float16')
                    add_out = conv_out + y
                    linear_out = self._linear(add_out)

            self.assertEqual(conv_out.dtype, paddle.float16)
            self.assertEqual(add_out.dtype, paddle.float16)
            self.assertEqual(linear_out.dtype, paddle.float16)


if __name__ == '__main__':
    unittest.main()
