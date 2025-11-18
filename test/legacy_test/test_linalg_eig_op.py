#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import paddle


class TestEigAPI0Size(unittest.TestCase):
    def test_errors(self):
        with paddle.device("cpu"):
            for shape in [[0, 0], [0, 4, 4], [1, 0, 2, 3, 3]]:
                x = paddle.randn(
                    shape=shape, dtype='float32', requires_grad=True
                )
                w, v = paddle.linalg.eig(x)
                self.assertEqual(w.shape, shape[:-1])
                self.assertEqual(v.shape, shape)

                (dw_dx,) = paddle.grad(w.abs().sum(), x, retain_graph=True)
                self.assertEqual(dw_dx.shape, x.shape)
                (dv_dx,) = paddle.grad(v.abs().sum(), x, retain_graph=True)
                self.assertEqual(dv_dx.shape, x.shape)
                (dwv_dx,) = paddle.grad(
                    w.abs().sum() + v.abs().sum(), x, retain_graph=True
                )
                self.assertEqual(dwv_dx.shape, x.shape)


if __name__ == '__main__':
    unittest.main()
