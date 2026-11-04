# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import paddle


def build_failing_graph():
    """Build a graph whose second order backward is doomed to fail.

    ``dout`` has an illegal shape (it should be [4, 0]): matmul_grad itself
    succeeds because of the 0-size fast return, but taking grad of its output
    runs MatmulDoubleGradNode, whose composite implementation calls the matmul
    forward api and fails the K dimension check in InferMeta.
    """
    x = paddle.randn([4, 8])
    y = paddle.randn([8, 0])
    dout = paddle.randn([4, 3])
    for tensor in (x, y, dout):
        tensor.stop_gradient = False
    dx, _ = paddle._C_ops.matmul_grad(x, y, dout, False, False)
    return [x, y, dout], dx


class TestGradEnabledRestoreOnBackwardError(unittest.TestCase):
    """A backward node raising an exception must not leak the global grad switch.

    Composite (prim) backward nodes turn grad recording off before running and
    turn it back on afterwards. When the composite implementation throws, the
    restore must still happen, otherwise the whole thread silently stops
    recording the computation graph.
    """

    def check_restored(self, run_backward):
        paddle.set_grad_enabled(True)
        inputs, dx = build_failing_graph()

        with self.assertRaises(ValueError):
            run_backward(inputs, dx)

        self.assertTrue(paddle.is_grad_enabled())
        probe = paddle.randn([2, 2])
        probe.stop_gradient = False
        self.assertFalse((probe * 2).stop_gradient)

    def test_paddle_grad_without_create_graph(self):
        def run(inputs, dx):
            paddle.grad(
                [dx],
                inputs,
                grad_outputs=[paddle.randn(dx.shape)],
                create_graph=False,
                allow_unused=True,
            )

        self.check_restored(run)

    def test_paddle_grad_with_create_graph(self):
        def run(inputs, dx):
            paddle.grad(
                [dx],
                inputs,
                grad_outputs=[paddle.randn(dx.shape)],
                create_graph=True,
                allow_unused=True,
            )

        self.check_restored(run)

    def test_tensor_backward(self):
        def run(inputs, dx):
            dx.backward(paddle.randn(dx.shape))

        self.check_restored(run)

    def test_tensor_backward_with_retain_graph(self):
        def run(inputs, dx):
            dx.backward(paddle.randn(dx.shape), retain_graph=True)

        self.check_restored(run)


if __name__ == '__main__':
    unittest.main()
