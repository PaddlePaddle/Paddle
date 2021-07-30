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

import unittest
import numpy as np
from paddle.fluid.tests.unittests.op_test import OpTest, skip_check_grad_ci
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
from paddle.fluid.framework import convert_np_dtype_to_dtype_

paddle.enable_static()


@skip_check_grad_ci(
    reason="reduce_max is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework.")
class TestNPUReduceMaxOp(OpTest):
    """Remove Max with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_max"
        self.set_npu()
        self.init_dtype()

        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {'dim': [-1]}
        self.outputs = {
            'Out': self.inputs['X'].max(axis=tuple(self.attrs['dim']))
        }

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def init_dtype(self):
        self.dtype = np.float32


@skip_check_grad_ci(
    reason="reduce_max is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework.")
class TestReduceMaxOpMultiAxises(TestNPUReduceMaxOp):
    """Remove Max with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_max"
        self.set_npu()
        self.init_dtype()

        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {'dim': [-2, -1]}
        self.outputs = {
            'Out': self.inputs['X'].max(axis=tuple(self.attrs['dim']))
        }


@skip_check_grad_ci(
    reason="reduce_max is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework.")
class TestReduceAll(TestNPUReduceMaxOp):
    """Remove Max with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_max"
        self.set_npu()
        self.init_dtype()

        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {'reduce_all': True}
        self.outputs = {'Out': self.inputs['X'].max()}


@skip_check_grad_ci(
    reason="reduce_max is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework.")
class TestReduceMaxOpWithOutDtype_bool(TestNPUReduceMaxOp):
    """Remove Max with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_max"
        self.set_npu()
        self.init_dtype()

        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {
            'dim': [-2, -1],
            'out_dtype': int(core.VarDesc.VarType.BOOL)
        }
        self.outputs = {
            'Out':
            self.inputs['X'].max(axis=tuple(self.attrs['dim'])).astype(np.bool)
        }


@skip_check_grad_ci(
    reason="reduce_max is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework.")
class TestReduceMaxOpWithOutDtype_int16(TestNPUReduceMaxOp):
    """Remove Max with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_max"
        self.set_npu()
        self.init_dtype()

        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {
            'dim': [-2, -1],
            'out_dtype': int(core.VarDesc.VarType.INT16)
        }

        self.out = self.inputs['X'].max(axis=tuple(self.attrs['dim']))

        self.outputs = {
            'Out':
            self.inputs['X'].max(axis=tuple(self.attrs['dim'])).astype(np.int16)
        }


@skip_check_grad_ci(
    reason="reduce_max is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework.")
class TestReduceMaxOpWithOutDtype_int32(TestNPUReduceMaxOp):
    """Remove Max with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_max"
        self.set_npu()
        self.init_dtype()

        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {
            'dim': [-2, -1],
            'out_dtype': int(core.VarDesc.VarType.INT32)
        }
        self.outputs = {
            'Out':
            self.inputs['X'].max(axis=tuple(self.attrs['dim'])).astype(np.int32)
        }


@skip_check_grad_ci(
    reason="reduce_max is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework.")
class TestReduceMaxOpWithOutDtype_int64(TestNPUReduceMaxOp):
    """Remove Max with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_max"
        self.set_npu()
        self.init_dtype()

        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {
            'dim': [-2, -1],
            'out_dtype': int(core.VarDesc.VarType.INT64)
        }
        self.outputs = {
            'Out':
            self.inputs['X'].max(axis=tuple(self.attrs['dim'])).astype(np.int64)
        }


@skip_check_grad_ci(
    reason="reduce_max is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework.")
class TestReduceMaxOpWithOutDtype_fp16(TestNPUReduceMaxOp):
    """Remove Max with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_max"
        self.set_npu()
        self.init_dtype()

        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {
            'dim': [-2, -1],
            'out_dtype': int(core.VarDesc.VarType.FP16)
        }

        self.out = self.inputs['X'].max(axis=tuple(self.attrs['dim']))

        self.outputs = {
            'Out': self.inputs['X'].max(
                axis=tuple(self.attrs['dim'])).astype(np.float16)
        }

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


@skip_check_grad_ci(
    reason="reduce_max is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework.")
class TestReduceMaxOpWithOutDtype_fp32(TestNPUReduceMaxOp):
    """Remove Max with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_max"
        self.set_npu()
        self.init_dtype()

        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {
            'dim': [-2, -1],
            'out_dtype': int(core.VarDesc.VarType.FP32)
        }
        self.outputs = {
            'Out': self.inputs['X'].max(
                axis=tuple(self.attrs['dim'])).astype(np.float32)
        }


@skip_check_grad_ci(
    reason="reduce_max is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework.")
class TestReduceMaxOpWithOutDtype_fp64(TestNPUReduceMaxOp):
    """Remove Max with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_max"
        self.set_npu()
        self.init_dtype()

        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {
            'dim': [-2, -1],
            'out_dtype': int(core.VarDesc.VarType.FP64)
        }
        self.outputs = {
            'Out': self.inputs['X'].max(
                axis=tuple(self.attrs['dim'])).astype(np.float64)
        }


@skip_check_grad_ci(
    reason="reduce_max is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework.")
class TestReduceMaxOpWithOutDtype_fp32_2(TestNPUReduceMaxOp):
    """Remove Max with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_max"
        self.set_npu()
        self.init_dtype()

        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {
            'dim': [-2, -1],
            'out_dtype': int(core.VarDesc.VarType.FP32)
        }
        self.outputs = {
            'Out': self.inputs['X'].max(
                axis=tuple(self.attrs['dim'])).astype(np.float32)
        }

    def init_dtype(self):
        self.dtype = np.float16


if __name__ == '__main__':
    unittest.main()
