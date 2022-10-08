#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import unittest
import sys

sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.framework import convert_np_dtype_to_dtype_, Program, program_guard

paddle.enable_static()


class SequenceMaskTestBase(OpTest):

    def set_npu(self):
        self.__class__.use_npu = True

    def initDefaultParameters(self):
        self.op_type = 'sequence_mask'
        self.maxlen = 10
        self.mask_dtype = 'int64'
        self.x = [[0, 3, 4], [5, 7, 9]]

    def initParameters(self):
        pass

    def setUp(self):
        self.set_npu()
        self.initDefaultParameters()
        self.initParameters()
        if not isinstance(self.x, np.ndarray):
            self.x = np.array(self.x)

        self.inputs = {'X': self.x}
        self.outputs = {'Y': self.calc_ground_truth_mask()}
        self.attrs = {
            'maxlen': self.maxlen,
            'out_dtype': convert_np_dtype_to_dtype_(self.mask_dtype)
        }

    def calc_ground_truth_mask(self):
        maxlen = np.max(self.x) if self.maxlen < 0 else self.maxlen
        shape = self.x.shape + (maxlen, )
        index_broadcast = np.broadcast_to(np.reshape(
            range(maxlen), newshape=[1] * self.x.ndim + [-1]),
                                          shape=shape)
        x_broadcast = np.broadcast_to(np.reshape(self.x,
                                                 newshape=self.x.shape +
                                                 (-1, )),
                                      shape=shape)
        return (index_broadcast < x_broadcast).astype(self.mask_dtype)

    def test_check_output(self):
        self.check_output_with_place(paddle.NPUPlace(0))


class SequenceMaskTest1(SequenceMaskTestBase):

    def initParameters(self):
        self.mask_dtype = 'bool'


class SequenceMaskTest2(SequenceMaskTestBase):

    def initParameters(self):
        self.mask_dtype = 'uint8'


class SequenceMaskTest3(SequenceMaskTestBase):

    def initParameters(self):
        self.mask_dtype = 'int32'


class SequenceMaskTest4(SequenceMaskTestBase):

    def initParameters(self):
        self.mask_dtype = 'float32'


class SequenceMaskTest5(SequenceMaskTestBase):

    def initParameters(self):
        self.mask_dtype = 'float64'


class SequenceMaskTest6(SequenceMaskTestBase):

    def initParameters(self):
        self.maxlen = -1


class SequenceMaskTestBase_tensor_attr(OpTest):

    def set_npu(self):
        self.__class__.use_npu = True

    def initDefaultParameters(self):
        self.op_type = 'sequence_mask'
        self.maxlen = 10
        self.maxlen_tensor = np.ones((1), 'int32') * 10
        self.mask_dtype = 'int64'
        self.x = [[0, 3, 4], [5, 7, 9]]

    def initParameters(self):
        pass

    def setUp(self):
        self.set_npu()
        self.initDefaultParameters()
        self.initParameters()
        if not isinstance(self.x, np.ndarray):
            self.x = np.array(self.x)

        self.inputs = {'X': self.x, 'MaxLenTensor': self.maxlen_tensor}
        self.outputs = {'Y': self.calc_ground_truth_mask()}
        self.attrs = {'out_dtype': convert_np_dtype_to_dtype_(self.mask_dtype)}

    def calc_ground_truth_mask(self):
        maxlen = np.max(self.x) if self.maxlen < 0 else self.maxlen
        shape = self.x.shape + (maxlen, )
        index_broadcast = np.broadcast_to(np.reshape(
            range(maxlen), newshape=[1] * self.x.ndim + [-1]),
                                          shape=shape)
        x_broadcast = np.broadcast_to(np.reshape(self.x,
                                                 newshape=self.x.shape +
                                                 (-1, )),
                                      shape=shape)
        return (index_broadcast < x_broadcast).astype(self.mask_dtype)

    def test_check_output(self):
        self.check_output()


class SequenceMaskTest1_tensor_attr(SequenceMaskTestBase_tensor_attr):

    def initParameters(self):
        self.mask_dtype = 'bool'


class SequenceMaskTest2_tensor_attr(SequenceMaskTestBase_tensor_attr):

    def initParameters(self):
        self.mask_dtype = 'uint8'


class SequenceMaskTest3_tensor_attr(SequenceMaskTestBase_tensor_attr):

    def initParameters(self):
        self.mask_dtype = 'int32'


class SequenceMaskTest4_tensor_attr(SequenceMaskTestBase_tensor_attr):

    def initParameters(self):
        self.mask_dtype = 'float32'


class SequenceMaskTest5_tensor_attr(SequenceMaskTestBase_tensor_attr):

    def initParameters(self):
        self.mask_dtype = 'float64'


class TestSequenceMaskOpError(unittest.TestCase):

    def test_errors(self):
        with program_guard(Program(), Program()):
            input_data = np.random.uniform(1, 5, [4]).astype("float32")

            def test_Variable():
                # the input must be Variable
                fluid.layers.sequence_mask(input_data, maxlen=4)

            self.assertRaises(TypeError, test_Variable)


if __name__ == '__main__':
    unittest.main()
