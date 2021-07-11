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

from __future__ import print_function

import numpy as np
import unittest
import sys
sys.path.append("..")
from op_test import OpTest, skip_check_grad_ci
import paddle

paddle.enable_static()

SEED = 2021
EPOCH = 100


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestDropoutOp(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.set_npu()
        self.init_dtype()
        self.inputs = {'X': np.random.random((32, 64)).astype(self.dtype)}
        self.attrs = {
            'dropout_prob': 0.0,
            'fix_seed': True,
            'is_test': False,
            'dropout_implementation': 'upscale_in_train'
        }
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': np.ones((32, 64)).astype('uint8')
        }

    def init_dtype(self):
        self.dtype = np.float32

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)

    def test_check_grad_normal(self):
        if self.dtype == np.float16:
            return
        self.check_grad_with_place(
            self.place, ['X'], 'Out', check_dygraph=False)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestDropoutOpInput1d(TestDropoutOp):
    # the input is 1-D
    def setUp(self):
        self.op_type = "dropout"
        self.set_npu()
        self.init_dtype()
        self.inputs = {'X': np.random.random((2000, )).astype(self.dtype)}
        self.attrs = {
            'dropout_prob': 0.0,
            'fix_seed': True,
            'is_test': False,
            'dropout_implementation': 'upscale_in_train'
        }
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': np.ones((2000)).astype('uint8')
        }


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestDropoutOp2(TestDropoutOp):
    # the dropout_prob is 1.0
    def setUp(self):
        self.op_type = "dropout"
        self.set_npu()
        self.init_dtype()
        self.inputs = {'X': np.random.random((32, 64)).astype(self.dtype)}
        self.attrs = {
            'dropout_prob': 1.0,
            'fix_seed': True,
            'is_test': False,
            'dropout_implementation': 'upscale_in_train'
        }
        self.outputs = {
            'Out': np.zeros((32, 64)).astype('float32'),
            'Mask': np.zeros((32, 64)).astype('uint8')
        }


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestDropoutOp3(TestDropoutOp):
    # the dropout_prob is 1.0
    def setUp(self):
        self.op_type = "dropout"
        self.set_npu()
        self.init_dtype()
        self.inputs = {'X': np.random.random((32, 64, 2)).astype(self.dtype)}
        self.attrs = {
            'dropout_prob': 0.0,
            'fix_seed': True,
            'is_test': False,
            'dropout_implementation': 'upscale_in_train'
        }
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': np.ones((32, 64, 2)).astype('uint8')
        }


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
@skip_check_grad_ci(reason="For inference, check_grad is not required.")
class TestDropoutOpInference(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.set_npu()
        self.init_dtype()
        self.inputs = {'X': np.random.random((32, 64)).astype(self.dtype)}
        self.attrs = {
            'dropout_prob': 0.35,
            'fix_seed': True,
            'is_test': True,
            'dropout_implementation': 'upscale_in_train'
        }
        self.outputs = {'Out': self.inputs['X']}

    def init_dtype(self):
        self.dtype = np.float32

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
@skip_check_grad_ci(reason="For inference, check_grad is not required.")
class TestDropoutOpInference2(TestDropoutOpInference):
    def setUp(self):
        self.op_type = "dropout"
        self.set_npu()
        self.init_dtype()
        self.inputs = {'X': np.random.random((32, 64, 3)).astype(self.dtype)}
        self.attrs = {
            'dropout_prob': 0.75,
            'is_test': True,
            'dropout_implementation': 'upscale_in_train'
        }
        self.outputs = {'Out': self.inputs['X']}


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestDropoutOpWithSeed(TestDropoutOp):
    # the input is 1-D
    def setUp(self):
        self.op_type = "dropout"
        self.set_npu()
        self.init_dtype()
        self.inputs = {
            "X": np.random.random((32, 64)).astype(self.dtype),
            "Seed": np.asarray(
                [125], dtype="int32")
        }
        self.attrs = {
            'dropout_prob': 0.0,
            'is_test': False,
            'dropout_implementation': 'upscale_in_train'
        }
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': np.ones((32, 64)).astype('uint8')
        }


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestDropoutOpFp16(TestDropoutOp):
    def init_dtype(self):
        self.dtype = np.float16

    def set_npu(self):
        self.__class__.use_npu = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.NPUPlace(0)


if __name__ == '__main__':
    unittest.main()
