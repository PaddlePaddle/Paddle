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

import paddle
from paddle import base

paddle.enable_static()


def sgd_wrapper(
    param, learning_rate, grad, master_param=None, multi_precision=False
):
    paddle._C_ops.sgd_(
        param, learning_rate, grad, master_param, multi_precision
    )


class TestSGDOpWithLargeInput(unittest.TestCase):
    def runTest(self):
        paddle.enable_static()
        data = paddle.tensor.fill_constant(shape=[1], value=128, dtype='int64')
        label = paddle.tensor.fill_constant(
            shape=[1, 150], value=0.5, dtype='float32'
        )
        emb = paddle.static.nn.embedding(
            input=data, size=(10000000, 150), dtype='float32'
        )
        out = paddle.nn.functional.normalize(x=emb, axis=-1)

        cost = paddle.nn.functional.square_error_cost(input=out, label=label)
        avg_cost = paddle.mean(cost)
        sgd_optimizer = paddle.optimizer.SGD(learning_rate=0.001)
        sgd_optimizer.minimize(avg_cost)

        place = base.CPUPlace()
        exe = base.Executor(place)
        exe.run(base.default_startup_program())
        compiled_prog = base.compiler.CompiledProgram(
            base.default_main_program()
        )
        result = exe.run(compiled_prog, fetch_list=[avg_cost])


class TestSGDV2(unittest.TestCase):
    def test_sgd(self):
        paddle.enable_static()

        def check_sgd_optimizer(optimizer_attr):
            init_program = paddle.static.Program()
            program = paddle.static.Program()
            block = program.global_block()
            mul_x = block.create_parameter(
                dtype="float32",
                shape=[5, 10],
                name="mul.x",
                optimize_attr=optimizer_attr,
            )
            mul_y = block.create_var(
                dtype="float32", shape=[10, 8], name="mul.y"
            )
            mul_out = block.create_var(
                dtype="float32", shape=[5, 8], name="mul.out"
            )
            mean_out = block.create_var(
                dtype="float32", shape=[1], name="mean.out"
            )
            block.append_op(
                type="mul",
                inputs={"X": mul_x, "Y": mul_y},
                outputs={"Out": mul_out},
                attrs={"x_num_col_dims": 1},
            )
            block.append_op(
                type="mean", inputs={"X": mul_out}, outputs={"Out": mean_out}
            )
            sgd_optimizer = paddle.optimizer.SGD(learning_rate=0.01)
            opts, _ = sgd_optimizer.minimize(mean_out, init_program)
            return opts

        opts = check_sgd_optimizer({'learning_rate': 1.1})
        self.assertEqual(len(opts), 2)
        self.assertEqual([op.type for op in opts], ["scale", "sgd"])

        opts = check_sgd_optimizer({'learning_rate': 1.0})
        self.assertEqual(len(opts), 1)
        self.assertEqual([op.type for op in opts], ["sgd"])


class TestSGDMultiPrecision2_0(unittest.TestCase):
    def dygraph_sgd_mp(self, mp):
        paddle.disable_static()
        paddle.seed(10)
        paddle.set_device('gpu')
        input = paddle.randn((2, 2))
        model = paddle.nn.Linear(2, 2)
        optimizer = paddle.optimizer.SGD(
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

    def static_sgd_mp(self, mp):
        paddle.enable_static()
        paddle.seed(10)
        np.random.seed(10)
        exe = paddle.static.Executor('gpu')
        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        optimizer = paddle.optimizer.SGD(multi_precision=mp)

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
                place=paddle.CUDAPlace(0), scope=paddle.static.global_scope()
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
        output1_dy, params1_dy = self.dygraph_sgd_mp(mp=True)
        output2_dy, params2_dy = self.dygraph_sgd_mp(mp=False)
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
        output1_st = self.static_sgd_mp(mp=True)
        output2_st = self.static_sgd_mp(mp=False)
        for idx in range(len(output1_st)):
            np.testing.assert_allclose(
                output1_st[idx].astype('float32'),
                output2_st[idx].astype('float32'),
                rtol=1e-05,
                atol=0.1,
            )


if __name__ == "__main__":
    unittest.main()
