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

<<<<<<< HEAD
=======
from __future__ import print_function

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import unittest
import numpy as np
from paddle.fluid.tests.unittests.op_test import OpTest, skip_check_grad_ci
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
from paddle.fluid.framework import convert_np_dtype_to_dtype_

paddle.enable_static()


class TestNPUReduceProd(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_npu()
        self.init_dtype()

        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {'dim': [0]}
        self.outputs = {
            'Out': self.inputs['X'].prod(axis=tuple(self.attrs['dim']))
        }

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def init_dtype(self):
        self.dtype = np.float32


class TestNPUReduceProd2(TestNPUReduceProd):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_npu()
        self.init_dtype()

        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {}  # default 'dim': [0]
        self.outputs = {'Out': self.inputs['X'].prod(axis=tuple([0]))}


class TestNPUReduceProd3(TestNPUReduceProd):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_npu()
        self.init_dtype()

        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        # self.attrs = {'dim': [0]}
        self.outputs = {'Out': self.inputs['X'].prod(axis=tuple([0]))}


class TestNPUReduceProd6D(TestNPUReduceProd):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_npu()
        self.init_dtype()

        self.inputs = {
            'X': np.random.random((5, 6, 2, 3, 4, 2)).astype(self.dtype)
        }
        self.attrs = {'dim': [2, 3, 4]}
        self.outputs = {
            'Out': self.inputs['X'].prod(axis=tuple(self.attrs['dim']))
        }


class TestNPUReduceProd8D(TestNPUReduceProd):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_npu()
        self.init_dtype()

        self.inputs = {
            'X': np.random.random((2, 5, 3, 2, 2, 3, 4, 2)).astype(self.dtype)
        }
        self.attrs = {'dim': [2, 3, 4]}
        self.outputs = {
            'Out': self.inputs['X'].prod(axis=tuple(self.attrs['dim']))
        }


class TestReduceAll(TestNPUReduceProd):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_npu()
        self.init_dtype()

        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {'reduce_all': True}
        self.outputs = {'Out': self.inputs['X'].prod()}


class TestNPUReduceProdWithOutDtype_bool(TestNPUReduceProd):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_npu()
        self.init_dtype()

        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {'dim': [0], 'out_dtype': int(core.VarDesc.VarType.BOOL)}
        self.outputs = {
<<<<<<< HEAD
            'Out': self.inputs['X']
            .prod(axis=tuple(self.attrs['dim']))
            .astype(np.bool_)
=======
            'Out':
            self.inputs['X'].prod(axis=tuple(self.attrs['dim'])).astype(
                np.bool_)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }


class TestNPUReduceProdWithOutDtype_int16(TestNPUReduceProd):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_npu()
        self.init_dtype()

        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {'dim': [0], 'out_dtype': int(core.VarDesc.VarType.INT16)}
        self.outputs = {
<<<<<<< HEAD
            'Out': self.inputs['X']
            .prod(axis=tuple(self.attrs['dim']))
            .astype(np.int16)
=======
            'Out':
            self.inputs['X'].prod(axis=tuple(self.attrs['dim'])).astype(
                np.int16)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }


class TestNPUReduceProdWithOutDtype_int32(TestNPUReduceProd):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_npu()
        self.init_dtype()

        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {'dim': [0], 'out_dtype': int(core.VarDesc.VarType.INT32)}
        self.outputs = {
<<<<<<< HEAD
            'Out': self.inputs['X']
            .prod(axis=tuple(self.attrs['dim']))
            .astype(np.int32)
=======
            'Out':
            self.inputs['X'].prod(axis=tuple(self.attrs['dim'])).astype(
                np.int32)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }


class TestNPUReduceProdWithOutDtype_int64(TestNPUReduceProd):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_npu()
        self.init_dtype()

        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {'dim': [0], 'out_dtype': int(core.VarDesc.VarType.INT64)}
        self.outputs = {
<<<<<<< HEAD
            'Out': self.inputs['X']
            .prod(axis=tuple(self.attrs['dim']))
            .astype(np.int64)
=======
            'Out':
            self.inputs['X'].prod(axis=tuple(self.attrs['dim'])).astype(
                np.int64)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }


class TestNPUReduceProdWithOutDtype_fp16(TestNPUReduceProd):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_npu()
        self.init_dtype()

        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {'dim': [0], 'out_dtype': int(core.VarDesc.VarType.FP16)}
        self.outputs = {
<<<<<<< HEAD
            'Out': self.inputs['X']
            .prod(axis=tuple(self.attrs['dim']))
            .astype(np.float16)
=======
            'Out':
            self.inputs['X'].prod(axis=tuple(self.attrs['dim'])).astype(
                np.float16)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestNPUReduceProdWithOutDtype_fp32(TestNPUReduceProd):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_npu()
        self.init_dtype()

        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {'dim': [0], 'out_dtype': int(core.VarDesc.VarType.FP32)}
        self.outputs = {
<<<<<<< HEAD
            'Out': self.inputs['X']
            .prod(axis=tuple(self.attrs['dim']))
            .astype(np.float32)
=======
            'Out':
            self.inputs['X'].prod(axis=tuple(self.attrs['dim'])).astype(
                np.float32)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }


class TestNPUReduceProdWithOutDtype_fp64(TestNPUReduceProd):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_npu()
        self.init_dtype()

        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {'dim': [0], 'out_dtype': int(core.VarDesc.VarType.FP64)}
        self.outputs = {
<<<<<<< HEAD
            'Out': self.inputs['X']
            .prod(axis=tuple(self.attrs['dim']))
            .astype(np.float64)
=======
            'Out':
            self.inputs['X'].prod(axis=tuple(self.attrs['dim'])).astype(
                np.float64)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }


@skip_check_grad_ci(reason="right now not implement grad op")
class TestNPUReduceProdWithOutDtype_fp32_2(TestNPUReduceProd):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_npu()
        self.init_dtype()

        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {'dim': [0], 'out_dtype': int(core.VarDesc.VarType.FP32)}
        self.outputs = {
<<<<<<< HEAD
            'Out': self.inputs['X']
            .prod(axis=tuple(self.attrs['dim']))
            .astype(np.float32)
=======
            'Out':
            self.inputs['X'].prod(axis=tuple(self.attrs['dim'])).astype(
                np.float32)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }

    def init_dtype(self):
        self.dtype = np.float16


if __name__ == '__main__':
    unittest.main()
