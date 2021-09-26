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

import unittest
import numpy as np
import paddle


def _jac_func()

def _compute_numerical_hessian(func, xs, delta, np_dtype):
    xs = _check_tensors(xs, "xs")
    ys = _check_tensors(func(*xs), "ys")
    fin_size = len(xs)
    hessian = list([] for _ in range(fin_size))
    for i in range(fin_size):
        hessian_i = list([] for _ in range(fin_size))
        for j in range(fin_size):
            hessian_i[j] = np.zeros(
                (_product(xs[i].shape), _product(xs[j].shape)), dtype=np_dtype)
        hessian[i] = hessian_i

    for i in range(fin_size):
        for p in range(_product(xs[i].shape)):
            orig_i = _get_item(xs[i], p)
            x_pos_i = orig_i + delta
            x_neg_i = orig_i - delta
            for j in range(fin_size):
                for q in range(_product(xs[j].shape)):
                    orig_j = _get_item(xs[j], q)
                    x_pos_j = orig_j + delta
                    x_neg_j = orig_j - delta
                    xs[i] = _set_item(xs[i], p, x_pos_i)


    for j in range(fin_size):
        for q in range(_product(xs[j].shape)):
            orig = _get_item(xs[j], q)
            x_pos = orig + delta
            xs[j] = _set_item(xs[j], q, x_pos)
            ys_pos = _check_tensors(func(*xs), "ys_pos")

            x_neg = orig - delta
            xs[j] = _set_item(xs[j], q, x_neg)
            ys_neg = _check_tensors(func(*xs), "ys_neg")

            xs[j] = _set_item(xs[j], q, orig)

            for i in range(fout_size):
                for p in range(_product(ys[i].shape)):
                    y_pos = _get_item(ys_pos[i], p)
                    y_neg = _get_item(ys_neg[i], p)
                    jacobian[i][j][p][q] = (y_pos - y_neg) / delta / 2.
    return hessian


class TestHessian(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.shape = (4, 4)
        self.dtype = 'float32'
        self.np_dtype = np.float32
        self.numerical_delta = 1e-4
        self.rtol = 1e-3
        self.atol = 1e-3
        self.x = paddle.rand(shape=self.shape, dtype=self.dtype)
        self.y = paddle.rand(shape=self.shape, dtype=self.dtype)

    def test_single_input(self):
        def func(x):
            return paddle.sum(paddle.matmul(x, x))

        self.x.stop_gradient = False
        hessian = paddle.autograd.hessian(func, self.x)
        print("hessian: ", hessian)


if __name__ == "__main__":
    unittest.main()
