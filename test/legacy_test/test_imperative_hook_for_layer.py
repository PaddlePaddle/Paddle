# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import get_places

import paddle
from paddle import base

call_forward_post_hook = False
call_forward_pre_hook = False


class SimpleNet(paddle.nn.Layer):
    def __init__(
        self,
        hidden_size,
        vocab_size,
        num_steps=20,
        init_scale=0.1,
        is_sparse=False,
        dtype='float32',
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.init_scale = init_scale
        self.num_steps = num_steps
        paddle.set_default_dtype(dtype)
        self.embedding = paddle.nn.Embedding(
            vocab_size,
            hidden_size,
            sparse=is_sparse,
            weight_attr=base.ParamAttr(
                name='embedding_para',
                initializer=paddle.nn.initializer.Uniform(
                    low=-init_scale, high=init_scale
                ),
            ),
        )
        self.softmax_bias = self.create_parameter(
            attr=base.ParamAttr(),
            shape=[self.vocab_size],
            dtype=dtype,
            default_initializer=paddle.nn.initializer.Uniform(
                low=-self.init_scale, high=self.init_scale
            ),
        )

    def forward(self, input, label):
        x_emb = self.embedding(input)
        projection = paddle.matmul(
            x_emb, paddle.transpose(self.embedding.weight, perm=[1, 0])
        )
        projection = paddle.add(projection, self.softmax_bias)
        projection = paddle.reshape(projection, shape=[-1, self.vocab_size])
        loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=projection, label=label, soft_label=False
        )
        loss = paddle.reshape(loss, shape=[-1, self.num_steps])
        loss = paddle.mean(loss, axis=[0])
        loss = paddle.sum(loss)

        return loss


def forward_post_hook(layer, input, output):
    global call_forward_post_hook
    call_forward_post_hook = True


def forward_pre_hook(layer, input):
    global call_forward_pre_hook
    call_forward_pre_hook = True


def forward_post_hook1(layer, input, output):
    return output * 2


def forward_pre_hook1(layer, input):
    input_return = (input[0] * 2, input[1])
    return input_return


def forward_pre_hook_with_kwargs(layer, args, kwargs):
    kwargs['x'] = kwargs['x'] * 2
    return (args, kwargs)


def forward_post_hook_with_kwargs(layer, inputs, kwargs, outputs):
    outputs = outputs + kwargs["x"]
    return outputs


class SimpleNetWithKWArgs(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = x + y
        return z


class DummyContextManager:
    def __init__(self, inp):
        self.input = inp

    def __enter__(self, *args, **kwargs):
        self.input.append(2)

    def __exit__(self, *args, **kwargs):
        self.input.append(-1)


class FailsNetInForward(paddle.nn.Layer):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, fail: bool = True):
        if fail:
            raise RuntimeError("failing in forward")
        return x


class Test_Forward_Hook(unittest.TestCase):
    # test forward_pre_hook and forward_post_hook that have return value
    def test_forward_hook_return_value(self):
        seed = 90

        for place in get_places():
            with base.dygraph.guard(place):
                paddle.seed(seed)
                base.set_flags({'FLAGS_sort_sum_gradient': True})

                input_word = (
                    np.array(
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8]
                    )
                    .reshape(6, 3)
                    .astype('int64')
                )
                input_word1 = input_word * 2
                input_word = input_word.reshape((-1, 3, 1))
                input_word1 = input_word1.reshape((-1, 3, 1))
                y_data = (
                    np.array(
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                    )
                    .reshape(6, 3)
                    .astype('int64')
                )
                y_data = y_data.reshape((-1, 1))

                input = paddle.to_tensor(input_word)
                input1 = paddle.to_tensor(input_word1)
                y = paddle.to_tensor(y_data)

                simplenet = SimpleNet(
                    hidden_size=20,
                    vocab_size=32,
                    num_steps=3,
                    init_scale=0.1,
                    is_sparse=False,
                    dtype="float32",
                )

                # origin, don't register any hook
                outs_origin = simplenet(input, y)
                outs_origin1 = simplenet(input1, y)

                # register forward_pre_hook
                forward_pre_hook_handle1 = simplenet.register_forward_pre_hook(
                    forward_pre_hook1
                )
                outs_pre_hook = simplenet(input, y)
                np.testing.assert_array_equal(
                    outs_pre_hook.numpy(), outs_origin1.numpy()
                )

                # remove forward_pre_hook
                forward_pre_hook_handle1.remove()
                outs_pre_hook = simplenet(input, y)
                np.testing.assert_array_equal(
                    outs_pre_hook.numpy(), outs_origin.numpy()
                )

                # register forward_posst_hook
                forward_post_hook_handle1 = (
                    simplenet.register_forward_post_hook(forward_post_hook1)
                )
                outs_forward_hook = simplenet(input, y)
                np.testing.assert_array_equal(
                    outs_forward_hook.numpy(), outs_origin.numpy() * 2
                )

                # remove forward_post_hook
                forward_post_hook_handle1.remove()
                outs_forward_hook = simplenet(input, y)
                np.testing.assert_array_equal(
                    outs_forward_hook.numpy(), outs_origin.numpy()
                )

    # test forward_pre_hook and forward_post_hook that don't have return value
    def test_forward_hook(self):
        seed = 90

        for place in get_places():
            with base.dygraph.guard(place):
                paddle.seed(seed)
                base.set_flags({'FLAGS_sort_sum_gradient': True})

                global call_forward_post_hook
                global call_forward_pre_hook

                input_word = (
                    np.array(
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8]
                    )
                    .reshape(6, 3)
                    .astype('int64')
                )
                input_word = input_word.reshape((-1, 3, 1))
                y_data = (
                    np.array(
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                    )
                    .reshape(6, 3)
                    .astype('int64')
                )
                y_data = y_data.reshape((-1, 1))

                input = paddle.to_tensor(input_word)
                y = paddle.to_tensor(y_data)

                simplenet = SimpleNet(
                    hidden_size=20,
                    vocab_size=32,
                    num_steps=3,
                    init_scale=0.1,
                    is_sparse=False,
                    dtype="float32",
                )

                # origin, don't register any hook
                outs_origin = simplenet(input, y)
                self.assertFalse(call_forward_post_hook)
                self.assertFalse(call_forward_pre_hook)

                # register forward_post_hook and forward_pre_hook
                forward_post_hook_handle = simplenet.register_forward_post_hook(
                    forward_post_hook
                )
                forward_pre_hook_handle = simplenet.register_forward_pre_hook(
                    forward_pre_hook
                )
                outs_hook = simplenet(input, y)
                self.assertTrue(call_forward_post_hook)
                self.assertTrue(call_forward_pre_hook)

                outs_hook = simplenet(input, y)
                self.assertTrue(call_forward_post_hook)
                self.assertTrue(call_forward_pre_hook)

                # remove forward_post_hook
                forward_post_hook_handle.remove()
                call_forward_post_hook = False
                call_forward_pre_hook = False
                outs_remove_forward_hook = simplenet(input, y)
                self.assertFalse(call_forward_post_hook)
                self.assertTrue(call_forward_pre_hook)

                # remove forward_pre_hook
                forward_pre_hook_handle.remove()
                call_forward_post_hook = False
                call_forward_pre_hook = False
                outs_remove_hook = simplenet(input, y)
                self.assertFalse(call_forward_post_hook)
                self.assertFalse(call_forward_pre_hook)

    def test_always_called_forward_hooks(self):
        x = paddle.ones((10, 10))
        stack = []
        ctx = None

        def setup_context():
            nonlocal ctx
            ctx = DummyContextManager(stack)

        def ctx_setup_hook(m, i):
            setup_context()
            ctx.__enter__()

        def ctx_setup_failure_hook(m, i):
            setup_context()
            ctx.__enter__()
            raise RuntimeError("failing in ctx setup")

        def ctx_shutdown_hook(m, i, o):
            ctx.__exit__()

        def ctx_shutdown_failure_hook(m, i, o):
            ctx.__exit__()
            raise RuntimeError("failing in ctx shutdown")

        def throw_hook(m, i, o):
            raise RuntimeError("failing in throw")

        net = FailsNetInForward()
        forward_pre_hook_handle = net.register_forward_pre_hook(ctx_setup_hook)
        forward_post_hook_handle = net.register_forward_post_hook(
            ctx_shutdown_hook, always_call=True
        )
        self.assertTrue(len(net._forward_post_hooks_always_called) == 1)

        # make sure always_called forward hook runs when model.forward raises RuntimeError
        with self.assertRaisesRegex(RuntimeError, "failing in forward"):
            net(x=x)
        self.assertEqual(stack, [2, -1])

        # make sure that always_called forward hook does not run twice if there is no error
        net(x, fail=False)
        self.assertEqual(stack, [2, -1, 2, -1])

        # make sure always_called forward hook runs when forward pre hook raises RuntimeError
        forward_pre_hook_handle.remove()
        net.register_forward_pre_hook(ctx_setup_failure_hook)
        with self.assertRaisesRegex(RuntimeError, "failing in ctx setup"):
            net(x, fail=False)
        self.assertEqual(stack, [2, -1, 2, -1, 2, -1])

        # make sure always_called hook runs when another always_called forward hook raises an error
        forward_post_hook_handle2 = net.register_forward_post_hook(
            throw_hook, prepend=True, always_call=True
        )

        # error raised should not be error of the forced hook
        with self.assertRaisesRegex(RuntimeError, "failing in ctx setup"):
            net(x, fail=False)
        self.assertEqual(stack, [2, -1, 2, -1, 2, -1, 2, -1])

        # make sure that always called forward hooks are properly removed
        forward_post_hook_handle.remove()
        forward_post_hook_handle2.remove()
        self.assertTrue(len(net._forward_post_hooks_always_called) == 0)

        # make sure that always called forward hook is not run twice if it fails while running
        forward_post_hook_handle3 = net.register_forward_post_hook(
            ctx_shutdown_failure_hook, always_call=True
        )
        with self.assertRaisesRegex(RuntimeError, "failing in ctx setup"):
            net(x, fail=False)
        self.assertEqual(stack, [2, -1, 2, -1, 2, -1, 2, -1, 2, -1])


class TestHookWithKWArgs(unittest.TestCase):
    def test_kwargs_hook(self):
        x = paddle.randn((2, 3))
        y = paddle.randn((2, 3))

        # 1. test forward pre hook
        net = SimpleNetWithKWArgs()
        remove_handler = net.register_forward_pre_hook(
            forward_pre_hook_with_kwargs, with_kwargs=True
        )

        out = net(x=x, y=y)
        np.testing.assert_allclose(out.numpy(), (x * 2 + y).numpy())

        remove_handler.remove()
        out = net(x=x, y=y)
        np.testing.assert_allclose(out.numpy(), (x + y).numpy())

        # 2. test forward pre and forward post hooks
        net = SimpleNetWithKWArgs()
        net.register_forward_post_hook(
            forward_post_hook_with_kwargs, with_kwargs=True
        )
        net.register_forward_pre_hook(
            forward_pre_hook_with_kwargs, with_kwargs=True
        )

        out = net(x=x, y=y)
        np.testing.assert_allclose(
            out.numpy(), (x * 4 + y).numpy(), rtol=1e-5, atol=1e-6
        )


if __name__ == '__main__':
    unittest.main()
