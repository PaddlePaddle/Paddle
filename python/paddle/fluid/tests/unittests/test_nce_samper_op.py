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


class TestNceSamplerOp(unittest.TestCase):
    def setUp(self):
        word_count = [10 for i in range(self.total_class_numbers)]
        word_count[0] = 5
        word_count[1] = 15

        self.dict_path = "./test_nce_sampler_op_word_count"
        self.total_class_numbers = 10
        self.num_neg_samples = 5

        with open(self.dict_path, 'w') as fout:
            for i in range(self.total_class_numbers):
                fout.write(word_count[i])

    def test_program(self):
        inputs = fluid.layers.data(shape=[2], dtype='int64_t', name='x')
        out = fluid.layers.nce_sampler(
            self.dict_path,
            self.num_total_classes,
            inputs,
            self.num_neg_samples,
            seed=0,
            factor=1.0)
        print(str(fluid.default_main_program()))
        print(str(fluid.default_startup_program()))


if __name__ == '__main__':
    unittest.main()
