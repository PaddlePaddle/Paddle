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

import paddle
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper, type_dict_str_to_numpy

paddle.enable_static()


class XPUTestScatterOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'scatter'
        self.use_dynamic_create_class = True

    def dynamic_create_class(self):
        base_class = self.TestScatterOp
        classes = []
        test_data_case = []

        # case1
        ref_np = np.ones((3, 50))
        index_np = np.array([1, 2])
        updates_np = np.random.random((2, 50))
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        data_dict = {
            'init_ref_np': ref_np,
            'init_index_np': index_np,
            'init_updates_np': updates_np,
            'init_output_np': output_np,
            'test_name': 'case1'
        }
        test_data_case.append(data_dict)

        # case2
        ref_np = np.ones((3, 3))
        index_np = np.array([1, 2])
        updates_np = np.random.random((2, 3))
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        data_dict = {
            'init_ref_np': ref_np,
            'init_index_np': index_np,
            'init_updates_np': updates_np,
            'init_output_np': output_np,
            'test_name': 'case2'
        }
        test_data_case.append(data_dict)

        # case3
        ref_np = np.ones((3, 3))
        zeros_np = np.zeros([2, 3])
        index_np = np.array([1, 1]).astype("int32")
        updates_np = np.random.randint(low=-1000, high=1000, size=(2, 3))
        output_np = np.copy(ref_np)
        output_np[index_np] = zeros_np
        for i in range(0, len(index_np)):
            output_np[index_np[i]] += updates_np[i]
        data_dict = {
            'init_ref_np': ref_np,
            'init_index_np': index_np,
            'init_updates_np': updates_np,
            'init_output_np': output_np,
            'test_name': 'case3'
        }
        test_data_case.append(data_dict)

        for data_dict in test_data_case:
            for index_type in ['int32', 'int64']:
                for overwrite in [True, False]:
                    class_name = 'XPUTestScatterOp_index_type_' + data_dict[
                        'test_name'] + '_' + str(index_type) + '_' + str(
                            overwrite)
                    attr_dict = data_dict
                    attr_dict['index_type'] = type_dict_str_to_numpy[index_type]
                    attr_dict['init_overwrite'] = overwrite
                    classes.append([class_name, attr_dict])
        return base_class, classes

    class TestScatterOp(XPUOpTest):
        def setUp(self):
            self.init_config()
            self.index_type = np.int32 if not hasattr(
                self, 'index_type') else self.index_type
            self.overwrite = True if not hasattr(
                self, 'init_overwrite') else self.init_overwrite

            if not hasattr(self, 'init_ref_np'):
                self.ref_np = np.ones((3, 50)).astype(self.dtype)
                self.index_np = np.array([1, 2]).astype(self.index_type)
                self.updates_np = np.random.random((2, 50)).astype(self.dtype)
                self.output_np = np.copy(self.ref_np)
                self.output_np[self.index_np] = self.updates_np
            else:
                self.ref_np = self.init_ref_np.astype(self.dtype)
                self.index_np = self.init_index_np.astype(self.index_type)
                self.updates_np = self.init_updates_np.astype(self.dtype)
                self.output_np = self.init_output_np.astype(self.dtype)

            self.inputs = {
                'X': self.ref_np,
                'Ids': self.index_np,
                'Updates': self.updates_np
            }
            self.attrs = {'overwrite': self.overwrite}
            self.outputs = {'Out': self.output_np}

        def init_config(self):
            self.op_type = "scatter"
            self.place = paddle.XPUPlace(0)
            self.dtype = self.in_type
            self.__class__.no_need_check_grad = True

        def test_check_output(self):
            self.check_output_with_place(self.place)


support_types = get_xpu_op_support_types('scatter')
for stype in support_types:
    create_test_class(globals(), XPUTestScatterOp, stype)

if __name__ == '__main__':
    unittest.main()
