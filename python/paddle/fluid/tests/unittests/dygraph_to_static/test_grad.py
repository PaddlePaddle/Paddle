# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import numpy as np
import paddle
import unittest


class GradLayer(paddle.nn.Layer):
    def __init__(self):
        super(GradLayer, self).__init__()

    @paddle.jit.to_static
    def forward(self):
        x = paddle.ones(shape=[1], dtype='float32')
        x.stop_gradient = False
        y = x * x
        dx = paddle.grad(outputs=[y], inputs=[x])[0]
        return dx


class TestGrad(unittest.TestCase):
    def _run(self, func, to_static):
        prog_trans = paddle.jit.ProgramTranslator()
        prog_trans.enable(to_static)
        return func().numpy()

    def test(self):
        grad_layer = GradLayer()
        static_res = self._run(grad_layer, to_static=False)
        dygraph_res = self._run(grad_layer, to_static=True)
        self.assertTrue(np.allclose(static_res, dygraph_res))


if __name__ == '__main__':
    unittest.main()
