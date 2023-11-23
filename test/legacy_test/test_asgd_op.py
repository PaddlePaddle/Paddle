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


def asgd_wrapper(
    param,
    learning_rate,
    grad,
    d,
    y,
    n,
    master_param=None,
    multi_precision=False,
):
    paddle._C_ops.asgd_(
        param,
        learning_rate,
        grad,
        d,
        y,
        n,
        None,
        False,
    )


class TestASGDOp(OpTest):
    def setUp(self):
        self.op_type = "asgd"
        self.python_api = asgd_wrapper
        self.python_out_sig = ['Out']
        self.conf()
        params = np.random.random((self.h, self.w)).astype("float32")
        learning_rate = np.array([0.1]).astype("float32")
        n = np.array([1000]).astype("float32")
        grads = np.random.random((self.h, self.w)).astype("float32")
        ds = np.random.random((self.h, self.w)).astype("float32")
        ys = np.random.random((self.h, self.w)).astype("float32")

        ds_out = ds - ys + grads
        ys_out = grads.copy()
        params_out = params - (learning_rate / n) * ds_out

        self.inputs = {
            "param": params,
            "learning_rate": learning_rate,
            "grad": grads,
            "d": ds,
            "y": ys,
            "n": n,
        }

        self.outputs = {
            "param_out": params_out,
            "d_out": ds_out,
            "y_out": ys_out,
        }

    def conf(self):
        self.h = 102
        self.w = 105

    def test_check_output(self):
        self.check_output(check_pir=True)


if __name__ == "__main__":
    unittest.main()
