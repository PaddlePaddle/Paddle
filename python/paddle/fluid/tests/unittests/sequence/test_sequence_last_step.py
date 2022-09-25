# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
from paddle.fluid.framework import convert_np_dtype_to_dtype_, Program, program_guard
import paddle.fluid.core as core
import numpy as np
import copy
import unittest
import sys

sys.path.append("../")
from op_test import OpTest


class TestSequenceLastStepOpError(unittest.TestCase):

    def test_errors(self):
        with program_guard(Program(), Program()):

            def test_Variable():
                # the input must be Variable
                input_data = np.random.randint(1, 5, [4]).astype("int64")
                fluid.layers.sequence_last_step(input_data)

            self.assertRaises(TypeError, test_Variable)

            def test_input_dtype():
                # the dtype of input must be int64
                type_data = fluid.layers.data(name='type_data',
                                              shape=[7, 1],
                                              append_batch_size=False,
                                              dtype='int64',
                                              lod_level=1)
                fluid.layers.sequence_last_step(type_data)

            self.assertRaises(TypeError, test_input_dtype)


if __name__ == '__main__':
    unittest.main()
