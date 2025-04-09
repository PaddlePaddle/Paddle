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

import contextlib
import os
import random
import unittest

import numpy as np

import paddle
from paddle import base, regularizer
from paddle.base import core


class TestL1Decay(unittest.TestCase):
    def test_l1decay_regularizer(self):
        with paddle.pir_utils.IrGuard():
            main_program = paddle.static.Program()
            with paddle.static.program_guard(main_program):
                block = main_program.global_block()
                mul_x = paddle.pir.core.create_parameter(
                    dtype="float32",
                    shape=[5, 10],
                    name="mul.x",
                    regularizer=regularizer.L1Decay(0.5),
                    initializer=paddle.nn.initializer.Constant(1),
                )
                self.assertIsNotNone(mul_x.regularizer)
                self.assertTrue(
                    isinstance(mul_x.regularizer, regularizer.L1Decay)
                )

                mul_y = paddle.static.data(
                    dtype="float32", shape=[10, 8], name="mul.y"
                )
                mul_out = paddle.matmul(mul_x, mul_y)
                mean_out = paddle.mean(mul_out)
                grads = paddle.autograd.ir_backward.grad(mean_out, [mul_x])
                params_grads = [(mul_x, grads[0])]
                self.assertEqual(len(params_grads), 1)
                count_ops = len(block.ops)
                optimizer = paddle.optimizer.Adam()
                params_grads = optimizer.append_regularization_ops(params_grads)
                self.assertEqual(len(params_grads), 1)
                self.assertEqual(len(block.ops), count_ops + 5)
                self.assertEqual(block.ops[-1].name(), 'pd_op.add_n')
                self.assertEqual(block.ops[-3].name(), 'pd_op.scale')
                self.assertEqual(block.ops[-5].name(), 'pd_op.sign')


class TestRegularizer(unittest.TestCase):
    def setUp(self):
        self.word_len = 1500
        self.train_data = [
            [(random.sample(range(1000), 10), [0])] for _ in range(2)
        ]

    def get_places(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(core.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        return places

    @contextlib.contextmanager
    def scope_prog_guard(self, main_prog, startup_prog):
        scope = base.core.Scope()
        with base.unique_name.guard():
            with base.scope_guard(scope):
                with base.program_guard(main_prog, startup_prog):
                    yield

    def run_program(self, place, feed_list):
        exe = base.Executor(place)
        feeder = base.DataFeeder(feed_list=feed_list, place=place)
        exe.run(base.default_startup_program())

        main_prog = base.default_main_program()
        param_list = [var.name for var in main_prog.block(0).all_parameters()]

        param_sum = []
        for data in self.train_data:
            out = exe.run(
                main_prog, feed=feeder.feed(data), fetch_list=param_list
            )
            p_sum = 0
            for v in out:
                p_sum += np.sum(np.abs(v))
            param_sum.append(p_sum)
        return param_sum

    def check_l2decay_regularizer(self, place, model):
        paddle.seed(1)
        paddle.framework.random._manual_program_seed(1)
        main_prog = base.framework.Program()
        startup_prog = base.framework.Program()
        with self.scope_prog_guard(
            main_prog=main_prog, startup_prog=startup_prog
        ):
            data = paddle.static.data(
                name="words", shape=[-1, 1], dtype="int64"
            )
            label = paddle.static.data(
                name="label", shape=[-1, 1], dtype="int64"
            )

            avg_cost = model(data, label, self.word_len)

            optimizer = paddle.optimizer.Adagrad(
                learning_rate=0.1,
                weight_decay=paddle.regularizer.L2Decay(1.0),
            )
            optimizer.minimize(avg_cost)
            param_sum = self.run_program(place, [data, label])
        return param_sum

    def check_l2decay(self, place, model):
        paddle.seed(1)
        paddle.framework.random._manual_program_seed(1)
        main_prog = base.framework.Program()
        startup_prog = base.framework.Program()

        with self.scope_prog_guard(
            main_prog=main_prog, startup_prog=startup_prog
        ):
            data = paddle.static.data(
                name="words", shape=[-1, 1], dtype="int64"
            )
            label = paddle.static.data(
                name="label", shape=[-1, 1], dtype="int64"
            )

            avg_cost_l2 = model(data, label, self.word_len)

            param_list = base.default_main_program().block(0).all_parameters()
            para_sum = []
            for para in param_list:
                para_mul = paddle.square(x=para)
                para_sum.append(paddle.sum(para_mul))
            avg_cost_l2 += paddle.add_n(para_sum) * 0.5

            optimizer = paddle.optimizer.Adagrad(learning_rate=0.1)
            optimizer.minimize(avg_cost_l2)
            param_sum = self.run_program(place, [data, label])
        return param_sum

    def test_repeated_regularization(self):
        l1 = paddle.regularizer.L1Decay(coeff=0.1)
        l2 = paddle.regularizer.L2Decay(coeff=0.01)
        fc_param_attr = paddle.ParamAttr(
            regularizer=paddle.regularizer.L1Decay()
        )
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.uniform([2, 2, 3])
            linear = paddle.nn.Linear(3, 5, weight_attr=fc_param_attr)
            out = linear(x)
            loss = paddle.sum(out)
            sgd = paddle.optimizer.SGD(learning_rate=0.1, weight_decay=l2)
            sgd.minimize(loss)
        with base.dygraph.guard():
            input = paddle.to_tensor(np.random.randn(3, 2).astype('float32'))
            paddle.seed(1)
            if paddle.framework.use_pir_api():
                with paddle.pir_utils.OldIrGuard():
                    paddle.framework.random._manual_program_seed(1)
            else:
                paddle.framework.random._manual_program_seed(1)

            linear1 = paddle.nn.Linear(
                2, 2, weight_attr=fc_param_attr, bias_attr=fc_param_attr
            )
            linear2 = paddle.nn.Linear(
                2, 2, weight_attr=fc_param_attr, bias_attr=fc_param_attr
            )

            loss1 = linear1(input)
            loss1.backward()
            # set l2 regularizer in optimizer, but l1 in base.ParamAttr

            paddle.optimizer.SGD(
                parameters=linear1.parameters(),
                learning_rate=1e-2,
                weight_decay=l2,
            ).minimize(loss1)
            # only set l1 in base.ParamAttr
            loss2 = linear2(input)
            loss2.backward()
            paddle.optimizer.SGD(
                parameters=linear2.parameters(), learning_rate=1e-2
            ).minimize(loss2)
            # they should both be applied by l1, and keep the same
            np.testing.assert_allclose(
                linear1.weight.numpy(),
                linear2.weight.numpy(),
                rtol=1e-05,
                err_msg='weight should use the regularization in base.ParamAttr!',
            )
            np.testing.assert_allclose(
                linear1.bias.numpy(),
                linear2.bias.numpy(),
                rtol=1e-05,
                err_msg='bias should use the regularization in base.ParamAttr!',
            )


if __name__ == '__main__':
    unittest.main()
