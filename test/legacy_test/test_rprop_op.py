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


def rprop_wrapper(
    param,
    grad,
    prev,
    learning_rate,
    master_param=None,
    learning_rate_range=np.array((1e-5, 50)).astype("float32"),
    etas=np.array((0.5, 1.2)).astype("float32"),
    multi_precision=False,
):
    paddle._C_ops.rprop_(
        param,
        grad,
        prev,
        learning_rate,
        None,
        learning_rate_range,
        etas,
        False,
    )


class TestRpropOp(OpTest):
    def setUp(self):
        self.op_type = "rprop"
        self.python_api = rprop_wrapper
        self.python_out_sig = ['Out']
        self.conf()
        params = np.random.random((self.h, self.w)).astype("float32")
        grads = np.random.random((self.h, self.w)).astype("float32")
        prevs = np.random.random((self.h, self.w)).astype("float32")
        learning_rates = np.random.random((self.h, self.w)).astype("float32")

        np.subtract(params, 0.5, out=params)
        np.multiply(params, 100, out=params)
        np.subtract(grads, 0.5, out=grads)
        np.multiply(grads, 100, out=grads)
        np.subtract(prevs, 0.5, out=prevs)
        np.multiply(prevs, 100, out=prevs)
        np.multiply(learning_rates, 100, out=learning_rates)

        learning_rate_min = 1e-5
        learning_rate_max = 50
        eta_negative = 0.5
        eta_positive = 1.2

        param_outs = params.copy()
        prev_outs = prevs.copy()
        learning_rate_outs = learning_rates.copy()

        for i, param in enumerate(params):
            grad = grads[i]
            prev = prevs[i]
            lr = learning_rate_outs[i]
            param_out = param_outs[i]
            prev_out = prev_outs[i]

            sign = np.sign(np.multiply(grad, prev))
            sign[np.greater(sign, 0)] = eta_positive
            sign[np.less(sign, 0)] = eta_negative
            sign[np.equal(sign, 0)] = 1
            np.multiply(lr, sign, out=lr)
            lr[np.less(lr, learning_rate_min)] = learning_rate_min
            lr[np.greater(lr, learning_rate_max)] = learning_rate_max

            grad = grad.copy()
            grad[np.equal(sign, eta_negative)] = 0

            learning_rate_outs[i] = lr
            param_outs[i] = np.subtract(
                param_out, np.multiply(np.sign(grad), lr)
            )
            prev_outs[i] = grad.copy()

        self.inputs = {
            "param": params,
            "grad": grads,
            "prev": prevs,
            "learning_rate": learning_rates,
            "learning_rate_range": np.array((1e-5, 50)).astype("float32"),
            "etas": np.array((0.5, 1.2)).astype("float32"),
        }

        self.outputs = {
            "param_out": param_outs,
            "prev_out": prev_outs,
            "learning_rate_out": learning_rate_outs,
        }

    def conf(self):
        self.h = 102
        self.w = 105

    def test_check_output(self):
        self.check_output(check_pir=True)


if __name__ == "__main__":
    unittest.main()
