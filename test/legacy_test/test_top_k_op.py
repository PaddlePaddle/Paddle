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
from op_test import OpTest

import paddle


def api_wrapper(x, k):
    return paddle._legacy_C_ops.top_k(x, "k", k)


class TestTopkOp(OpTest):
    def setUp(self):
        self.variable_k = False
        self.set_args()
        self.op_type = "top_k"
        self.python_api = api_wrapper
        self.dtype = np.float64
        self.check_cinn = True
        self.init_dtype()

        k = self.top_k
        input = np.random.random((self.row, k)).astype(self.dtype)
        output = np.ndarray((self.row, k))
        indices = np.ndarray((self.row, k)).astype("int64")
        self.inputs = {'X': input}

        if self.variable_k:
            self.inputs['K'] = np.array([k]).astype("int32")
        else:
            self.attrs = {'k': k}

        for rowid in range(self.row):
            row = input[rowid]
            output[rowid] = np.sort(row)[::-1][:k]
            indices[rowid] = row.argsort()[::-1][:k]

        self.outputs = {'Out': output, 'Indices': indices}

    def init_dtype(self):
        pass

    def set_args(self):
        self.row = 100
        self.top_k = 1

    def test_check_output(self):
        self.check_output(check_cinn=self.check_cinn)

    def test_check_grad(self):
        self.check_grad({'X'}, 'Out', check_cinn=self.check_cinn)


class TestTopkOutAPI(unittest.TestCase):
    def test_out_in_dygraph(self):
        paddle.disable_static()
        x = paddle.to_tensor(
            np.array([[1, 4, 5, 7], [2, 6, 2, 5]]).astype('float32'),
            stop_gradient=False,
        )
        k = 2

        def run_case(case):
            out_values = paddle.zeros_like(x[:, :k])
            out_indices = paddle.zeros([x.shape[0], k], dtype='int64')
            out_values.stop_gradient = False
            out_indices.stop_gradient = False

            if case == 'return':
                values, indices = paddle.topk(x, k)
            elif case == 'input_out':
                paddle.topk(x, k, out=(out_values, out_indices))
                values, indices = out_values, out_indices
            elif case == 'both_return':
                values, indices = paddle.topk(
                    x, k, out=(out_values, out_indices)
                )
            elif case == 'both_input_out':
                _ = paddle.topk(x, k, out=(out_values, out_indices))
                values, indices = out_values, out_indices
            else:
                raise AssertionError

            ref_values, ref_indices = paddle._C_ops.topk(x, k, -1, True, True)
            np.testing.assert_allclose(
                values.numpy(), ref_values.numpy(), rtol=1e-6, atol=1e-6
            )
            np.testing.assert_allclose(
                indices.numpy(), ref_indices.numpy(), rtol=1e-6, atol=1e-6
            )

            loss = (values.mean() + indices.float().mean()).mean()
            loss.backward()
            return values.numpy(), indices.numpy(), x.grad.numpy()

        # run four scenarios
        v1, i1, g1 = run_case('return')
        x.clear_gradient()
        v2, i2, g2 = run_case('input_out')
        x.clear_gradient()
        v3, i3, g3 = run_case('both_return')
        x.clear_gradient()
        v4, i4, g4 = run_case('both_input_out')

        np.testing.assert_allclose(v1, v2, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(v1, v3, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(v1, v4, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(i1, i2, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(i1, i3, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(i1, i4, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(g1, g2, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(g1, g3, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(g1, g4, rtol=1e-6, atol=1e-6)

        paddle.enable_static()


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
