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

paddle.enable_static()


def rprop_wrapper(
    param,
    grad,
    prev,
    learning_rate,
    delta_min,
    delta_max,
    eta_negative,
    eta_positive,
    master_param=None,
    multi_precision=False,
):
    paddle._C_ops.rprop_(
        param,
        grad,
        prev,
        learning_rate,
        master_param,
        delta_min,
        delta_max,
        eta_negative,
        eta_positive,
        multi_precision,
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
        delta_min = 1e-6
        delta_max = 50
        eta_negative = 0.5
        eta_positive = 1.2

        param_outs = params.clone()
        prev_outs = prevs.clone()
        learning_rate_outs = learning_rates.clone()
        for i, param in enumerate(params):
            grad = grads[i]
            prev = prevs[i]
            lr = learning_rate_outs[i]
            param_out = param_outs[i]
            prev_out = prev_outs[i]
            sign = grad.mul(prev).sign()
            sign[sign.gt(0)] = eta_positive
            sign[sign.lt(0)] = eta_negative
            sign[sign.eq(0)] = 1

            lr.mul_(sign).clamp_(delta_min, delta_max)

            grad = grad.clone()
            grad[sign.eq(eta_negative)] = 0

            param_out.addcmul_(grad.sign(), lr, value=-1)
            prev_out.copy_(grad)

        self.inputs = {
            "Param": params,
            "Grad": grads,
            'Prev': prevs,
            "LearningRate": learning_rates,
        }

        self.outputs = {
            "ParamOut": param_outs,
            "PrevOut": prev_outs,
            "LearningRateOut": learning_rate_outs,
        }

        self.attrs = {
            'delta_min': delta_min,
            'delta_max': delta_max,
            'eta_negative': eta_negative,
            'eta_positive': eta_positive,
        }

    def conf(self):
        self.h = 102
        self.w = 105

    def test_check_output(self):
        self.check_output(check_pir=True)


if __name__ == "__main__":
    unittest.main()
