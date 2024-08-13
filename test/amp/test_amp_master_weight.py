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
from paddle.base import core


class SimpleNet(paddle.nn.Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        weight_attr = paddle.ParamAttr(
            name="weight", initializer=paddle.nn.initializer.Constant(value=0.5)
        )
        bias_attr = paddle.ParamAttr(
            name="bias", initializer=paddle.nn.initializer.Constant(value=1.0)
        )
        self.linear = paddle.nn.Linear(
            input_size, output_size, weight_attr, bias_attr
        )

    def forward(self, x):
        x = self.linear(x)
        return x


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
class TestMasterWeight(AmpTestBase):
    def run_dygraph(self, dtype, level, use_promote, max_iters, x_data):
        losses = []
        model = SimpleNet(100, 100)
        optimizer = paddle.optimizer.AdamW(
            learning_rate=0.01,
            parameters=model.parameters(),
        )
        scaler = paddle.amp.GradScaler()
        model, optimizer = paddle.amp.decorate(
            models=model,
            optimizers=optimizer,
            level=level,
            dtype=dtype,
        )

        for i in range(max_iters):
            with paddle.amp.auto_cast(
                enable=True,
                dtype=dtype,
                level=level,
                use_promote=use_promote,
            ):
                x = paddle.to_tensor(x_data, dtype='float16')
                out = model(x)
                loss = paddle.mean(out)
                losses.append(loss)
            scaled = scaler.scale(loss)
            scaled.backward()
            scaler.minimize(optimizer, scaled)
            optimizer.clear_grad()
        return losses

    def run_pir(self, dtype, level, use_promote, max_iters, x_data):
        with paddle.pir_utils.IrGuard():
            losses = []
            startup = paddle.static.Program()
            main = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                model = SimpleNet(100, 100)
                optimizer = paddle.optimizer.AdamW(
                    learning_rate=0.01,
                    parameters=model.parameters(),
                )
                scaler = paddle.amp.GradScaler(enable=True)
                model, optimizer = paddle.amp.decorate(
                    models=model,
                    optimizers=optimizer,
                    level=level,
                    dtype=dtype,
                )
                with paddle.amp.auto_cast(
                    enable=True,
                    dtype=dtype,
                    level=level,
                    use_promote=use_promote,
                ):
                    x = paddle.static.data('x', x_data.shape, 'float16')
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
            for iter_id in range(max_iters):
                results = exe.run(
                    main,
                    feed={'x': x_data},
                    fetch_list=[loss],
                )

                losses.append(results[0])

            return losses

    def run_static(self, dtype, level, use_promote, max_iters, x_data):
        paddle.enable_static()
        with paddle.pir_utils.OldIrGuard():
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            losses = []
            with paddle.utils.unique_name.guard():
                with paddle.static.program_guard(main_program, startup_program):
                    model = SimpleNet(100, 100)
                    optimizer = paddle.optimizer.AdamW(learning_rate=0.01)
                    optimizer = paddle.static.amp.decorate(
                        optimizer,
                        level=level,
                        dtype=dtype,
                        use_promote=use_promote,
                        master_weight=True,
                    )
                    x = paddle.static.data(
                        name='input', shape=[100, 100], dtype='float16'
                    )
                    out = model(x)
                    loss = paddle.mean(out)
                    optimizer.minimize(loss)

            if paddle.is_compiled_with_cuda():
                place = paddle.CUDAPlace(0)
            elif paddle.device.is_compiled_with_xpu():
                place = paddle.device.XPUPlace(0)
            else:
                raise ValueError("Only support CUDA or XPU Place.")
            exe = paddle.static.Executor(place)
            exe.run(startup_program)
            optimizer.amp_init(
                place,
                scope=paddle.static.global_scope(),
                rewrite_master_weight=True,
            )
            for iter_id in range(max_iters):
                results = exe.run(
                    program=main_program,
                    feed={x.name: x_data},
                    fetch_list=[loss],
                )
                print(
                    f"-- [AMP {dtype} {level}] iter={iter_id}, loss={results[0]}"
                )
                losses.append(results[0])

        paddle.disable_static()
        return losses

    def test_master_weight(self):
        np.random.seed(1)
        paddle.seed(1)
        dtype = 'float16'
        level = 'O2'
        use_promote = True
        total_steps = 4
        x_data = np.random.random(size=[100, 100]).astype("float16")

        loss_dygraph = self.run_dygraph(
            dtype, level, use_promote, total_steps, x_data
        )
        loss_static = self.run_static(
            dtype, level, use_promote, total_steps, x_data
        )
        loss_pir = self.run_pir(dtype, level, use_promote, total_steps, x_data)

        for i in range(total_steps):
            self.assertEqual(loss_dygraph[i], loss_static[i])
            self.assertEqual(loss_dygraph[i], loss_pir[i])


if __name__ == '__main__':
    unittest.main()
