# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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


def api_wrapper(x, label, ignore_index=-100):
    return paddle._legacy_C_ops.cross_entropy2(
        x, label, "ignore_index", ignore_index
    )


class CrossEntropy2OpTestBase(OpTest):
    def initParameters(self):
        return [32, 64], 'float64', -100, False

    def calc_output(self, logits, label, ignore_index):
        ret = np.zeros(shape=label.shape, dtype=logits.dtype)
        match_x = np.zeros(shape=label.shape, dtype=logits.dtype)
        for idx in range(label.shape[0]):
            if label[idx] == ignore_index:
                continue
            match_x[idx] = logits[idx][label[idx]]
            ret[idx] = -np.log(match_x[idx])
        return ret, match_x

    def setUp(self):
        (
            self.shape,
            self.dtype,
            self.ignore_index,
            self.drop_last_dim,
        ) = self.initParameters()
        self.op_type = 'cross_entropy2'
        self.python_api = api_wrapper
        self.python_out_sig = ["Y"]
        feature_size = int(self.shape[-1])
        batch_size = int(np.prod(self.shape) / feature_size)
        logits = (np.random.random(size=self.shape) + 1).astype(self.dtype)
        label_shape = (
            self.shape[0:-1] if self.drop_last_dim else self.shape[0:-1] + [1]
        )
        label = np.random.random_integers(
            low=0, high=feature_size - 1, size=label_shape
        ).astype('int64')
        outputs, match_x = self.calc_output(
            np.reshape(logits, [batch_size, feature_size]),
            np.reshape(label, [batch_size, 1]),
            self.ignore_index,
        )
        self.inputs = {'X': logits, 'Label': label}
        out_shape = label_shape
        self.outputs = {
            'Y': np.reshape(outputs, out_shape),
            'MatchX': np.reshape(match_x, self.shape[:-1] + [1]),
            'XShape': np.zeros(shape=logits.shape, dtype=logits.dtype),
        }
        self.attrs = {'ignore_index': self.ignore_index}

    def test_check_output(self):
        self.check_output(no_check_set=['XShape'])

    def test_check_grad(self):
        self.check_grad(
            inputs_to_check=['X'],
            output_names=['Y'],
            no_grad_set=['XShape', 'MatchX', 'Label'],
        )


class CrossEntropy2OpTest2(CrossEntropy2OpTestBase):
    def initParameters(self):
        return [32, 64], 'float64', 3, False


class CrossEntropy2OpTest2RemoveLastDim(CrossEntropy2OpTestBase):
    def initParameters(self):
        return [32, 64], 'float64', 3, True


class CrossEntropy2OpTest3(CrossEntropy2OpTestBase):
    def initParameters(self):
        return [4, 8, 16, 32], 'float64', -100, False


class CrossEntropy2OpTest3RemoveLastDim(CrossEntropy2OpTestBase):
    def initParameters(self):
        return [4, 8, 16, 32], 'float64', -100, True


class CrossEntropy2OpTest4(CrossEntropy2OpTestBase):
    def initParameters(self):
        return [4, 8, 16, 32], 'float64', 3, False


if __name__ == '__main__':
    unittest.main()
