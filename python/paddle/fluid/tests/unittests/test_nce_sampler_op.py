# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import paddle.fluid as fluid
from paddle.fluid.executor import Executor
import numpy as np


class TestNceSamplerOp(unittest.TestCase):
    def setUp(self):
        self.dict_path = "test_nce_sampler_op_word_count"
        self.total_class_numbers = 10
        self.num_neg_samples = 5

        word_count = [10 for i in range(self.total_class_numbers)]
        word_count[0] = 5
        word_count[1] = 15

        with open(self.dict_path, 'w') as fout:
            for i in range(self.total_class_numbers):
                fout.write(str(word_count[i]) + "\n")

    def test_program(self, use_cuda=False):
        self.setUp()
        inputs = fluid.layers.data(
            shape=[2], dtype='int64', name='x', append_batch_size=False)
        out = fluid.layers.nce_sampler(
            self.dict_path,
            self.total_class_numbers,
            self.num_neg_samples,
            seed=0,
            factor=1.0,
            positive_inputs=inputs)

        custom_probs = np.array([0.1] * 10).astype(np.float32)
        custom_alias = np.array([-1] * 10).astype(int)
        custom_alias_probs = np.array([1.0] * 10).astype(np.float32)

        custom_probs[0] = 0.05
        custom_probs[1] = 0.15
        custom_alias[0] = 1
        custom_alias_probs[0] = 0.5

        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = Executor(place)

        exe.run(fluid.default_startup_program())
        self.assertTrue((np.array(fluid.global_scope().find_var(
            "CustomDistProbs").get_tensor()) == custom_probs).all())
        self.assertTrue((np.array(fluid.global_scope().find_var(
            "CustomDistAlias").get_tensor()) == custom_alias).all())
        self.assertTrue((np.array(fluid.global_scope().find_var(
            "CustomDistAliasProbs").get_tensor()) == custom_alias_probs).all())

        input_val = np.array(range(2)).astype(np.int64)
        result = exe.run(fluid.default_main_program(),
                         feed={"x": input_val},
                         fetch_list=[out])[0]
        self.assertEqual(len(result.shape), 1)
        self.assertEqual(result.shape[0], self.num_neg_samples)
        for item in result:
            self.assertTrue(item > 1 and item < 10)


if __name__ == '__main__':
    unittest.main()
