# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle
from paddle.base import core

paddle.enable_static()


class XPUTestUniqueOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = "unique"
        self.use_dynamic_create_class = False

    class TestUniqueOp(XPUOpTest):
        def setUp(self):
            self.op_type = "unique"
            self.init_dtype()
            self.init_config()

        def init_dtype(self):
            self.dtype = self.in_type

        def init_config(self):
            self.inputs = {
                'X': np.array([2, 3, 3, 1, 5, 3], dtype=self.dtype),
            }
            self.attrs = {
                'dtype': int(core.VarDesc.VarType.INT32),
                'return_index': True,
                'return_inverse': True,
                'is_sorted': True,  # is_sorted must be set to true to call paddle.unique rather than base.layers.unique
            }
            self.outputs = {
                'Out': np.array([1, 2, 3, 5], dtype=self.dtype),
                'Indices': np.array([3, 0, 1, 4], dtype='int32'),
                'Index': np.array([1, 2, 2, 0, 3, 2]),
            }

        def test_check_output(self):
            self.check_output_with_place(paddle.XPUPlace(0))

    class TestOne(TestUniqueOp):
        def init_config(self):
            self.inputs = {
                'X': np.array([2], dtype=self.dtype),
            }
            self.attrs = {
                'dtype': int(core.VarDesc.VarType.INT32),
                'return_index': True,
                'return_inverse': True,
                'is_sorted': True,
            }
            self.outputs = {
                'Out': np.array([2], dtype=self.dtype),
                'Indices': np.array([0], dtype='int32'),
                'Index': np.array([0], dtype='int32'),
            }

    class TestRandom(TestUniqueOp):
        def init_config(self):
            self.inputs = {
                'X': (np.random.random([150]) * 100.0).astype(self.dtype)
            }
            self.attrs = {
                'dtype': int(core.VarDesc.VarType.INT64),
                'return_index': True,
                'return_inverse': True,
                'return_counts': True,
                'is_sorted': True,
            }
            np_unique, np_index, reverse_index, np_counts = np.unique(
                self.inputs['X'],
                True,
                True,
                True,
            )

            self.outputs = {
                'Out': np_unique,
                'Indices': np_index,
                'Index': reverse_index,
                'Counts': np_counts,
            }

    class TestRandom2(TestUniqueOp):
        def init_config(self):
            self.inputs = {
                'X': (np.random.random([4, 7, 10]) * 100.0).astype(self.dtype)
            }
            unique, indices, inverse, counts = np.unique(
                self.inputs['X'],
                return_index=True,
                return_inverse=True,
                return_counts=True,
                axis=None,
            )
            if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
                inverse = inverse.flatten()
            self.attrs = {
                'dtype': int(core.VarDesc.VarType.INT64),
                "return_index": True,
                "return_inverse": True,
                "return_counts": True,
                "axis": None,
                "is_sorted": True,
            }
            self.outputs = {
                'Out': unique,
                'Indices': indices,
                "Index": inverse,
                "Counts": counts,
            }

    class TestEmpty(TestUniqueOp):
        def init_config(self):
            self.inputs = {'X': np.ones([0, 4], dtype=self.dtype)}
            self.attrs = {
                'dtype': int(core.VarDesc.VarType.INT64),
                'return_index': True,
                'return_inverse': True,
                'return_counts': True,
                'is_sorted': True,
            }
            self.outputs = {
                'Out': np.ones([0], dtype=self.dtype),
                'Indices': np.ones([0], dtype=self.dtype),
                'Index': np.ones([0], dtype=self.dtype),
                'Counts': np.ones([0], dtype=self.dtype),
            }

    class TestUniqueOpAxis1(TestUniqueOp):
        def init_config(self):
            self.inputs = {
                'X': (np.random.random([3, 8, 8]) * 100.0).astype(self.dtype)
            }
            unique, indices, inverse, counts = np.unique(
                self.inputs['X'],
                return_index=True,
                return_inverse=True,
                return_counts=True,
                axis=1,
            )
            if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
                inverse = inverse.flatten()
            self.attrs = {
                'dtype': int(core.VarDesc.VarType.INT32),
                "return_index": True,
                "return_inverse": True,
                "return_counts": True,
                "axis": [1],
                "is_sorted": True,
            }
            self.outputs = {
                'Out': unique,
                'Indices': indices,
                "Index": inverse,
                "Counts": counts,
            }

    class TestUniqueOpAxis2(TestUniqueOp):
        def init_config(self):
            self.inputs = {
                'X': (np.random.random([1, 10]) * 100.0).astype(self.dtype)
            }

            unique, indices, inverse, counts = np.unique(
                self.inputs['X'],
                return_index=True,
                return_inverse=True,
                return_counts=True,
                axis=0,
            )
            if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
                inverse = inverse.flatten()
            self.attrs = {
                'dtype': int(core.VarDesc.VarType.INT32),
                "return_index": True,
                "return_inverse": True,
                "return_counts": True,
                "axis": [0],
                "is_sorted": True,
            }

            self.outputs = {
                'Out': unique,
                'Indices': indices,
                "Index": inverse,
                "Counts": counts,
            }

    class TestUniqueOpAxisNeg(TestUniqueOp):
        def init_config(self):
            self.inputs = {
                'X': (np.random.random([6, 1, 8]) * 100.0).astype(self.dtype)
            }
            unique, indices, inverse, counts = np.unique(
                self.inputs['X'],
                return_index=True,
                return_inverse=True,
                return_counts=True,
                axis=-1,
            )
            if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
                inverse = inverse.flatten()
            self.attrs = {
                'dtype': int(core.VarDesc.VarType.INT32),
                "return_index": True,
                "return_inverse": True,
                "return_counts": True,
                "axis": [-1],
                "is_sorted": True,
            }
            self.outputs = {
                'Out': unique,
                'Indices': indices,
                "Index": inverse,
                "Counts": counts,
            }


support_types = get_xpu_op_support_types("unique")
for stype in support_types:
    create_test_class(globals(), XPUTestUniqueOp, stype)


if __name__ == "__main__":
    unittest.main()
