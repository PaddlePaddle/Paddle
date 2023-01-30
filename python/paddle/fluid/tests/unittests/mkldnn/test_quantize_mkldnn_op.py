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

<<<<<<< HEAD
import unittest

import numpy as np

import paddle
from paddle.fluid.tests.unittests.op_test import OpTest


class TestQuantizeOp(OpTest):
=======
from __future__ import print_function

import unittest
import numpy as np
from paddle.fluid.tests.unittests.op_test import OpTest
import paddle


class TestQuantizeOp(OpTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = 'quantize'
        self.scale = 255.0
        self.shift = 0.0
        self.input_size = [1, 1, 5, 5]  # Naive nChw16c
        self.is_negative = False
        self.output_format = 'NCHW'
        self.set_scale()
        self.set_shift()
        self.set_is_negative()
        self.set_input_size()
        self.set_output_format()
        self.prepare_input()
        self.prepare_output()

    def prepare_input(self):
        if self.is_negative:
            # input data values are from interval [-1.0, 1.0)
<<<<<<< HEAD
            self.input = (
                2 * np.random.random_sample(self.input_size) - 1
            ).astype('float32')
        else:
            # input data values are from interval [0.0, 1.0)
            self.input = (np.random.random_sample(self.input_size)).astype(
                'float32'
            )
=======
            self.input = (2 * np.random.random_sample(self.input_size) -
                          1).astype('float32')
        else:
            # input data values are from interval [0.0, 1.0)
            self.input = (np.random.random_sample(
                self.input_size)).astype('float32')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.inputs = {'Input': OpTest.np_dtype_to_fluid_dtype(self.input)}
        self.attrs = {
            'Scale': self.scale,
            'Shift': self.shift,
            'is_negative_input': self.is_negative,
<<<<<<< HEAD
            'output_format': self.output_format,
=======
            'output_format': self.output_format
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }

    def prepare_output(self):
        input_data_type = 'int8' if self.is_negative else 'uint8'
<<<<<<< HEAD
        output = np.rint(self.input * self.scale + self.shift).astype(
            input_data_type
        )
=======
        output = np.rint(self.input * self.scale +
                         self.shift).astype(input_data_type)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.outputs = {'Output': output}

    def test_check_output(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.check_output(check_dygraph=False)

    def check_raise_error(self, msg):
        try:
            self.check_output()
        except Exception as e:
            if msg in str(e):
                raise AttributeError
            else:
                print(e)

    def set_scale(self):
        pass

    def set_shift(self):
        pass

    def set_is_negative(self):
        pass

    def set_input_size(self):
        pass

    def set_output_format(self):
        pass


class TestQuantizeOp1(TestQuantizeOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_scale(self):
        self.scale = 127.0

    def set_is_negative(self):
        self.is_nagative = True


class TestQuantizeOp2(TestQuantizeOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_scale(self):
        self.scale = 255.0

    def set_is_negative(self):
        self.is_nagative = False


# 2-dim input
# P - positive input
class TestQuantizeOpShift_NCHW_2_P(TestQuantizeOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_output_format(self):
        self.output_format = 'NCHW'

    def set_is_negative(self):
        self.is_nagative = False

    def set_scale(self):
        self.scale = 255.0

    def set_shift(self):
        self.shift = 0.0

    def set_input_size(self):
        self.input_size = [2, 3]


# 2-dim input
# N - negative input
class TestQuantizeOpShift_NCHW_2_N(TestQuantizeOpShift_NCHW_2_P):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_is_negative(self):
        self.is_nagative = True

    def set_scale(self):
        self.scale = 127.0

    def set_shift(self):
        self.shift = 128.0


class TestQuantizeOpShift_NHWC_2_P(TestQuantizeOpShift_NCHW_2_P):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_output_format(self):
        self.output_format = 'NHWC'


class TestQuantizeOpShift_NHWC_2_N(TestQuantizeOpShift_NCHW_2_N):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_output_format(self):
        self.output_format = 'NHWC'


# 3-dim input
class TestQuantizeOpShift_NCHW_3_P(TestQuantizeOpShift_NCHW_2_P):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_input_size(self):
        self.input_size = [2, 3, 4]


class TestQuantizeOpShift_NCHW_3_N(TestQuantizeOpShift_NCHW_2_N):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_input_size(self):
        self.input_size = [2, 3, 4]


class TestQuantizeOpShift_NHWC_3_P(TestQuantizeOpShift_NCHW_3_P):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_output_format(self):
        self.output_format = 'NHWC'


class TestQuantizeOpShift_NHWC_3_N(TestQuantizeOpShift_NCHW_3_N):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_output_format(self):
        self.output_format = 'NHWC'


# 4-dim input
class TestQuantizeOpShift_NCHW_4_P(TestQuantizeOpShift_NCHW_2_P):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_input_size(self):
        self.input_size = [2, 3, 4, 5]


class TestQuantizeOpShift_NCHW_4_N(TestQuantizeOpShift_NCHW_2_N):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_input_size(self):
        self.input_size = [2, 3, 4, 5]


class TestQuantizeOpShift_NHWC_4_P(TestQuantizeOpShift_NCHW_4_P):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_output_format(self):
        self.output_format = 'NHWC'


class TestQuantizeOpShift_NHWC_4_N(TestQuantizeOpShift_NCHW_4_N):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_output_format(self):
        self.output_format = 'NHWC'


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
