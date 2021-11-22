#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import unittest
import numpy as np
from op_test import OpTest, skip_check_grad_ci


def rrelu_np(x: np.array,
             lower_bound: float=0.125,
             upper_bound: float=0.3333,
             is_test: bool=False):
    """

    """
    x = x.astype(np.float32)
    if is_test:
        middle_value = (lower_bound + upper_bound) / 2.0
        mask = copy.deepcopy(x)
        mask[x >= 0.0] = 1.0
        mask[x < 0.0] = middle_value
    else:
        x_shape = x.shape
        x = x.reshape(-1)
        mask = copy.deepcopy(x)
        for i in range(x.shape[0]):
            if x[i].item() >= 0.0:
                mask[i] = 1.0
            else:
                mask[i] = np.random.uniform(lower_bound, upper_bound)
        x = x.reshape(x_shape)
        mask = mask.reshape(x_shape)

    out = x * mask
    return out, mask


class TestRReLUOp(OpTest):
    def setUp(self):
        self.op_type = "rrelu"
        X = np.random.uniform(low=-100, high=10, size=(32, )).astype("float32")
        lower_bound = 0.0
        upper_bound = 0.5
        fix_seed = True
        seed = 100
        is_test = False
        self.inputs = {'X': X}
        self.attrs = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'fix_seed': fix_seed,
            'seed': seed,
            'is_test': is_test
        }
        np.random.seed(seed)
        Out, Mask = rrelu_np(
            x=X,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            is_test=is_test)
        self.outputs = {'Out': Out, 'Mask': Mask}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out')


class TestRReLUOp2(TestRReLUOp):
    def setUp(self):
        self.op_type = "rrelu"
        X = np.random.uniform(low=-100, high=10, size=(8, 16)).astype("float32")
        lower_bound = 0.4
        upper_bound = 0.99
        fix_seed = True
        seed = 3
        is_test = False
        self.inputs = {'X': X}
        self.attrs = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'fix_seed': fix_seed,
            'seed': seed,
            'is_test': is_test
        }
        np.random.seed(seed)
        Out, Mask = rrelu_np(
            x=X,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            is_test=is_test)
        self.outputs = {'Out': Out, 'Mask': Mask}


class TestRReLUOp3(TestRReLUOp):
    def setUp(self):
        self.op_type = "rrelu"
        X = np.random.uniform(
            low=-100, high=10, size=(8, 16, 32)).astype("float32")
        lower_bound = 0.5
        upper_bound = 0.51
        fix_seed = True
        seed = 5
        is_test = False
        self.inputs = {'X': X}
        self.attrs = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'fix_seed': fix_seed,
            'seed': seed,
            'is_test': is_test
        }
        np.random.seed(seed)
        Out, Mask = rrelu_np(
            x=X,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            is_test=is_test)
        self.outputs = {'Out': Out, 'Mask': Mask}


@skip_check_grad_ci(reason="For inference, check_grad is not required.")
class TestRReLUOp4(OpTest):
    def setUp(self):
        self.op_type = "rrelu"
        X = np.random.uniform(low=-100, high=10, size=(32, )).astype("float32")
        lower_bound = 0.0
        upper_bound = 0.3
        fix_seed = True
        seed = 11
        is_test = True
        self.inputs = {'X': X}
        self.attrs = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'fix_seed': fix_seed,
            'seed': seed,
            'is_test': is_test
        }
        Out, Mask = rrelu_np(
            x=X,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            is_test=is_test)
        self.outputs = {'Out': Out}

    def test_check_output(self):
        self.check_output()


@skip_check_grad_ci(reason="For inference, check_grad is not required.")
class TestRReLUOp5(OpTest):
    def setUp(self):
        self.op_type = "rrelu"
        X = np.random.uniform(
            low=-100, high=10, size=(32, 16, 8)).astype("float32")
        lower_bound = 0.0
        upper_bound = 0.3
        is_test = True
        self.inputs = {'X': X}
        self.attrs = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'is_test': is_test
        }
        Out, Mask = rrelu_np(
            x=X,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            is_test=is_test)
        self.outputs = {'Out': Out}

    def test_check_output(self):
        self.check_output()


class TestRReLUOpWithSeed(OpTest):
    def setUp(self):
        self.op_type = "rrelu"
        X = np.random.uniform(
            low=-100, high=10, size=(32, 16)).astype("float32")
        Seed = np.asarray([125], dtype="int32")
        lower_bound = 0.0
        upper_bound = 0.3
        is_test = False
        self.inputs = {'X': X, 'Seed': Seed}
        self.attrs = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'is_test': is_test
        }
        np.random.seed(125)
        Out, Mask = rrelu_np(
            x=X,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            is_test=is_test)
        self.outputs = {'Out': Out, 'Mask': Mask}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out', max_relative_error=0.05)


if __name__ == "__main__":
    unittest.main()
