# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from dygraph_to_static_utils import (
    Dy2StTestBase,
    test_ast_only,
    test_legacy_and_pt,
    test_pir_only,
)
from test_fetch_feed import Linear

import paddle

SEED = 2020


def nested_input(x, y):
    sum_res = x + y[0]

    z_elem = y[3]['z']
    sub_res = z_elem[0] - z_elem[1]

    mul_res = y[-1]['d']['da'] * y[-1]['d']['dc']
    mean_func = paddle.mean
    out = mean_func(sub_res) + mean_func(sum_res) + mean_func(mul_res)
    return out


def nested_output(x, y):
    sum_res = x + y
    sub_res = x - y
    mul_res = x * y

    out = {}
    out['z'] = sum_res
    out['a'] = [sub_res, 64, [mul_res, "cmd"]]
    return out


def fake_data(shape):
    x_data = np.random.random(shape).astype('float32')
    return paddle.to_tensor(x_data)


class TestWithNestedInput(Dy2StTestBase):
    def setUp(self):
        self.x = None
        self.y = None

    def fake_input(self):
        self.x = fake_data([10, 16])
        self.y = [
            fake_data([10, 16]),
            "preprocess_cmd",
            64,
            {
                'z': [fake_data([10, 12]), fake_data([10, 12])],
                'c': fake_data([10, 10]),
                'd': {'da': 12, 'dc': fake_data([10, 10])},
            },
        ]

    def _run(self, to_static):
        if self.x is None or self.y is None:
            self.fake_input()

        if to_static:
            out = paddle.jit.to_static(nested_input, full_graph=True)(
                self.x, self.y
            )
        else:
            out = nested_input(self.x, self.y)

        return out.numpy()

    def test_nest(self):
        dygraph_res = self._run(to_static=False)
        static_res = self._run(to_static=True)
        np.testing.assert_allclose(dygraph_res, static_res, rtol=1e-05)


class TestWithNestedOutput(Dy2StTestBase):
    def setUp(self):
        self.x = None
        self.y = None

    def _run(self, to_static):
        if self.x is None or self.y is None:
            self.x = fake_data([10, 16])
            self.y = fake_data([10, 16])

        if to_static:
            out = paddle.jit.to_static(nested_output, full_graph=True)(
                self.x, self.y
            )
        else:
            out = nested_output(self.x, self.y)

        return out

    def test_nest(self):
        dygraph_res = self._run(to_static=False)
        dygraph_res = paddle.utils.flatten(dygraph_res)

        static_res = self._run(to_static=True)
        static_res = paddle.utils.flatten(static_res)

        self.assertTrue(len(dygraph_res) == len(static_res))

        for dy_var, st_var in zip(dygraph_res, static_res):
            if isinstance(dy_var, paddle.Tensor):
                np.testing.assert_allclose(
                    dy_var.numpy(), st_var.numpy(), rtol=1e-05
                )
            else:
                self.assertTrue(dy_var, st_var)


class TestWithTrainAndEval(Dy2StTestBase):
    @test_ast_only
    @test_legacy_and_pt
    def test_legacy_ir_switch_eval_and_train(self):
        # TODO(cleanup-legacy-ir): Remove this test case
        linear_net = Linear()
        linear_net = paddle.jit.to_static(linear_net, full_graph=True)
        x_data = np.random.random((4, 10)).astype('float32')
        x = paddle.to_tensor(x_data)
        linear_net(x)

        _, train_partial_layer = linear_net.forward.program_cache.last()[-1]
        # check default mode is for training
        self.assertEqual(
            train_partial_layer.program, train_partial_layer._train_program
        )

        # switch to run test program after `eval()`
        linear_net.eval()
        linear_net(x)
        _, eval_partial_layer = linear_net.forward.program_cache.last()[-1]
        self.assertEqual(
            eval_partial_layer.program, eval_partial_layer._infer_program
        )

        # switch back into training
        linear_net.train()
        linear_net(x)
        self.assertEqual(
            train_partial_layer.program, train_partial_layer._train_program
        )

    @test_ast_only
    @test_pir_only
    def test_switch_eval_and_train(self):
        linear_net = Linear()
        linear_net = paddle.jit.to_static(linear_net, full_graph=True)
        x_data = np.random.random((4, 10)).astype('float32')
        x = paddle.to_tensor(x_data)
        linear_net(x)

        _, train_partial_layer = linear_net.forward.program_cache.last()[-1]
        # check default mode is for training
        self.assertEqual(
            train_partial_layer.program,
            train_partial_layer.train_program,
        )

        # switch to run test program after `eval()`
        linear_net.eval()
        linear_net(x)
        _, eval_partial_layer = linear_net.forward.program_cache.last()[-1]
        self.assertEqual(
            eval_partial_layer.program, eval_partial_layer.infer_program
        )

        # switch back into training
        linear_net.train()
        linear_net(x)
        self.assertEqual(
            train_partial_layer.program, train_partial_layer.train_program
        )


class TestWithNoGrad(Dy2StTestBase):
    @test_ast_only
    @test_legacy_and_pt
    def test_legacy_ir_with_no_grad(self):
        # TODO(cleanup-legacy-ir): Remove this test case
        linear_net = Linear()
        linear_net = paddle.jit.to_static(linear_net, full_graph=True)
        x_data = np.random.random((5, 10)).astype('float32')
        x = paddle.to_tensor(x_data)

        with paddle.no_grad():
            linear_net.train()
            linear_net(x)
            # BUG: 我们希望这里 是 ASTStaticFunction(StaticFunction):
            _, partial_layer = linear_net.forward.program_cache.last()[-1]
            self.assertEqual(
                partial_layer.program, partial_layer._train_program
            )

    @test_ast_only
    @test_pir_only
    def test_with_no_grad(self):
        linear_net = Linear()
        linear_net = paddle.jit.to_static(linear_net, full_graph=True)
        x_data = np.random.random((5, 10)).astype('float32')
        x = paddle.to_tensor(x_data)

        with paddle.no_grad():
            linear_net.train()
            linear_net(x)
            # BUG: 我们希望这里 是 ASTStaticFunction(StaticFunction):
            _, partial_layer = linear_net.forward.program_cache.last()[-1]
            self.assertEqual(partial_layer.program, partial_layer.train_program)


class GPT2LMHeadModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.embedding0 = paddle.nn.Embedding(20, 16)
        self.embedding1 = paddle.nn.Embedding(20, 32)
        self.lm_head_weight = paddle.to_tensor(
            np.random.rand(2, 3).astype('float32')
        )

    def forward(self, x):
        x = paddle.reshape(x, shape=[-1, 6])
        x1, x2, x3 = paddle.split(x=x, axis=1, num_or_sections=3)
        return x1


class TestPruneUnusedParamInProgram(Dy2StTestBase):
    def test_prune(self):
        input_ids = np.array([[15, 11, 6, 3, 18, 13]]).astype("float32")

        model = paddle.jit.to_static(GPT2LMHeadModel())
        model.eval()
        input_ids = paddle.to_tensor(input_ids)
        out = model(input_ids)
        np.testing.assert_array_equal(out.numpy(), [[15, 11]])


if __name__ == '__main__':
    unittest.main()
