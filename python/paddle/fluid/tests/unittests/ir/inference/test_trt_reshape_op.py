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

<<<<<<< HEAD
import unittest

import numpy as np
from inference_pass_test import InferencePassTest

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.static.nn as nn
from paddle.fluid.core import AnalysisConfig, PassVersionChecker


class TRTReshapeTest(InferencePassTest):
=======
from __future__ import print_function

import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import PassVersionChecker
from paddle.fluid.core import AnalysisConfig


class TRTReshapeTest(InferencePassTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.bs = 1
        self.input_shape = [16, 3, 8]
        self.reshape = [-1, 4, 4, 24]
        self.data_shape = [
<<<<<<< HEAD
            self.bs,
            self.input_shape[0],
            self.input_shape[1],
            self.input_shape[2],
        ]
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='data', shape=self.data_shape, dtype='float32'
            )
            reshape_out = self.append_reshape(data, self.reshape)
            out = nn.batch_norm(reshape_out, is_test=True)
=======
            self.bs, self.input_shape[0], self.input_shape[1],
            self.input_shape[2]
        ]
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name='data',
                              shape=self.data_shape,
                              dtype='float32')
            reshape_out = self.append_reshape(data, self.reshape)
            out = fluid.layers.batch_norm(reshape_out, is_test=True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.feeds = {
            'data': np.random.random(self.data_shape).astype('float32'),
        }
        self.enable_trt = True
        self.trt_parameters = TRTReshapeTest.TensorRTParam(
<<<<<<< HEAD
            1 << 30, self.bs, 1, AnalysisConfig.Precision.Float32, False, False
        )
        self.fetch_list = [out]

    def append_reshape(self, data, reshape):
        return paddle.reshape(data, reshape)
=======
            1 << 30, self.bs, 1, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def append_reshape(self, data, reshape):
        return fluid.layers.reshape(data, reshape)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
<<<<<<< HEAD
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class TRTReshapeTest1(TRTReshapeTest):
=======
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TRTReshapeTest1(TRTReshapeTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.bs = 2
        self.input_shape = [23, 13, 12]
        self.reshape = [2, 0, -1, 6]
        self.data_shape = [
<<<<<<< HEAD
            self.bs,
            self.input_shape[0],
            self.input_shape[1],
            self.input_shape[2],
        ]
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='data', shape=self.data_shape, dtype='float32'
            )
            reshape_out = self.append_reshape(data, self.reshape)
            out = nn.batch_norm(reshape_out, is_test=True)
=======
            self.bs, self.input_shape[0], self.input_shape[1],
            self.input_shape[2]
        ]
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name='data',
                              shape=self.data_shape,
                              dtype='float32')
            reshape_out = self.append_reshape(data, self.reshape)
            out = fluid.layers.batch_norm(reshape_out, is_test=True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.feeds = {
            'data': np.random.random(self.data_shape).astype('float32'),
        }
        self.enable_trt = True
        self.trt_parameters = TRTReshapeTest.TensorRTParam(
<<<<<<< HEAD
            1 << 30, self.bs, 1, AnalysisConfig.Precision.Float32, False, False
        )
=======
            1 << 30, self.bs, 1, AnalysisConfig.Precision.Float32, False, False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.fetch_list = [out]


class TRTReshapeTest2(TRTReshapeTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.bs = 2
        self.input_shape = [23, 13, 12]
        self.reshape = [2, 0, -1, 6]
        self.data_shape = [
<<<<<<< HEAD
            self.bs,
            self.input_shape[0],
            self.input_shape[1],
            self.input_shape[2],
        ]
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='data', shape=self.data_shape, dtype='float32'
            )
            reshape_out = paddle.reshape(x=data, shape=self.reshape)
            out = nn.batch_norm(reshape_out, is_test=True)
=======
            self.bs, self.input_shape[0], self.input_shape[1],
            self.input_shape[2]
        ]
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name='data',
                              shape=self.data_shape,
                              dtype='float32')
            reshape_out = fluid.layers.reshape(x=data, shape=self.reshape)
            out = fluid.layers.batch_norm(reshape_out, is_test=True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.feeds = {
            'data': np.random.random(self.data_shape).astype('float32')
        }
        self.enable_trt = True
        self.trt_parameters = TRTReshapeTest.TensorRTParam(
<<<<<<< HEAD
            1 << 30, self.bs, 1, AnalysisConfig.Precision.Float32, False, False
        )
=======
            1 << 30, self.bs, 1, AnalysisConfig.Precision.Float32, False, False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.fetch_list = [out]


class TRTReshapeTest3(TRTReshapeTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.bs = 1
        self.input_shape = [7, 16, 27]
        self.reshape = [1, 8, 14, 0]
        self.data_shape = [
<<<<<<< HEAD
            self.bs,
            self.input_shape[0],
            self.input_shape[1],
            self.input_shape[2],
        ]
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='data', shape=self.data_shape, dtype='float32'
            )
            bn_out = nn.batch_norm(data, is_test=True)
=======
            self.bs, self.input_shape[0], self.input_shape[1],
            self.input_shape[2]
        ]
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name='data',
                              shape=self.data_shape,
                              dtype='float32')
            bn_out = fluid.layers.batch_norm(data, is_test=True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            out = self.append_reshape(bn_out, self.reshape)
        self.feeds = {
            'data': np.random.random(self.data_shape).astype('float32'),
        }
        self.enable_trt = True
        self.trt_parameters = TRTReshapeTest.TensorRTParam(
<<<<<<< HEAD
            1 << 30, self.bs, 1, AnalysisConfig.Precision.Float32, False, False
        )
=======
            1 << 30, self.bs, 1, AnalysisConfig.Precision.Float32, False, False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        '''
        self.dynamic_shape_params = TRTReshapeTest.DynamicShapeParam({
            'data': [1, 3, 8, 8]
        }, {'data': [5, 100, 100, 100]}, {'data': [1, 3, 16, 16]}, False)
        '''
        self.fetch_list = [out]


if __name__ == "__main__":
    unittest.main()
