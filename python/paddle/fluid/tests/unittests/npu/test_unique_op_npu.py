#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import time
import unittest
import numpy as np
import sys
sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

paddle.enable_static()

class TestUniqueNPUOp(OpTest):
    def setUp(self):
        self.op_type = "unique"
        self.set_npu()
        self.init_config()

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def init_config(self):
        self.inputs = {'X': np.array([2, 3, 3, 1, 5, 3], dtype='int64')}
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT32),
                      'is_sorted': True,
                      'return_inverse': True,
                      'return_counts': True}
        self.outputs = {
            'Out': np.array(
                [2, 3, 1, 5], dtype='int64'),
            'Index': np.array(
                [0, 1, 1, 2, 3, 1], dtype='int32'),
            'Counts': np.array(
                [1, 3, 1, 1], dtype='int32')
        }

    def test_check_output(self):
        self.check_output_with_place(self.place)


if __name__ == "__main__":
    time.sleep(6000)
    unittest.main()


# class TestNPUUniqueOp(OpTest):
#     def setUp(self):
#         self.op_type = "unique"
#         self.set_npu()
#         self.init_config()

#     def set_npu(self):
#         self.__class__.use_npu = True
#         self.place = paddle.NPUPlace(0)

#     def test_check_output(self):
#         self.check_output_with_place(self.place, check_dygraph=False)

#     def init_config(self):
#         self.inputs = {'X': np.array([2, 3, 3, 1, 5, 3], dtype='int64'), }
#         self.attrs = {
#             'dtype': int(core.VarDesc.VarType.INT32),
#             "return_inverse": True,
#             "return_counts": True}
#         self.outputs = {
#             'Out': np.array(
#                 [2, 3, 1, 5], dtype='int64'),
#             'Index': np.array(
#                 [0, 1, 1, 2, 3, 1], dtype='int32')
#         }

    # def init_config(self):
    #     self.inputs = {'X': np.array([2, 3, 3, 1, 5, 3], dtype='int64')}
    #     self.attrs = {
    #         'dtype': int(core.VarDesc.VarType.INT32),
    #         # "return_index": True,
    #         "return_inverse": True,
    #         "return_counts": True,
    #         # "is_sorted": True
    #     }
    #     unique, indices, inverse, count = np.unique(
    #         self.inputs['X'],
    #         return_index=True,
    #         return_inverse=True,
    #         return_counts=True,
    #         axis=None
    #     )
    #     unique = np.array([2,3,1,5], dtype='int64')
    #     inverse = np.array([0,1,1,2,3,1], dtype='int64')
    #     self.outputs = {
    #         'Out': unique,
    #         # 'Indices': indices,
    #         "Index": inverse,
    #         # "Counts": count,
    #     }


#     def init_config(self):
#         self.inputs = {'X': np.array([2, 3, 3, 1, 5, 3], dtype='int64'), }
#         self.attrs = {'dtype': int(core.VarDesc.VarType.INT32)}
#         self.outputs = {
#             'Out': np.array(
#                 [2, 3, 1, 5], dtype='int64'),
#             'Index': np.array(
#                 [0, 1, 1, 2, 3, 1], dtype='int32')
#         }


# class TestNPUSortedUniqueOp(TestNPUUniqueOp):
#     def init_config(self):
#         self.inputs = {'X': np.array([2, 3, 3, 1, 5, 3], dtype='int64')}
#         self.attrs = {
#             'dtype': int(core.VarDesc.VarType.INT32),
#             "return_index": True,
#             "return_inverse": True,
#             "return_counts": True,
#             "is_sorted": True
#         }
#         unique, indices, inverse, count = np.unique(
#             self.inputs['X'],
#             return_index=True,
#             return_inverse=True,
#             return_counts=True,
#             axis=None
#         )
#         self.outputs = {
#             'Out': unique,
#             'Indices': indices,
#             "Index": inverse,
#             "Counts": count,
#         }


# class TestNPUUniqueAPI(unittest.TestCase):
#     def setUp(self):
#         self.__class__.use_npu = True
#         self.place = paddle.NPUPlace(0)

#     def test_static_graph(self):
#         train_prog = fluid.Program()
#         startup = fluid.Program()
#         with fluid.program_guard(train_prog, startup):
#             x = fluid.data(name='x', shape=[6], dtype='float64')
#             unique, indices, inverse, counts = paddle.unique(x, return_inverse=True, return_counts=True)

#             exe = fluid.Executor(self.place)
#             exe.run(startup)

#             x_np = np.array([2, 3, 3, 1, 5, 3]).astype('float64')
#             result = exe.run(train_prog, 
#                              feed={'x': x_np}, 
#                              fetch_list=[unique, indices, inverse, counts])

#         np_unique, np_indices, np_inverse, np_counts = np.unique(x_np, return_index=True, return_inverse=True, return_counts=True)
#         print(result[0], np_unique)
#         self.assertTrue(np.allclose(result[0], np_unique))
#         self.assertTrue(np.allclose(result[1], np_indices))
#         self.assertTrue(np.allclose(result[2], np_inverse))
#         self.assertTrue(np.allclose(result[3], np_counts))



