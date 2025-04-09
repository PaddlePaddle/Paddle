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

import unittest

import numpy as np
from op_test import OpTest

import paddle
from paddle.base import core


@unittest.skipIf(
    not core.supports_bfloat16(), "place does not support BF16 evaluation"
)
@unittest.skipIf(
    core.is_compiled_with_cuda(),
    "core is compiled with CUDA which has no BF implementation",
)
class TestSplitSectionsBF16OneDNNOp(OpTest):
    def init_data(self):
        self.x = np.random.random((4, 5, 6)).astype("uint16")
        self.axis = 1
        self.sections = [2, 1, 2]
        indices_or_sections = [2, 3]  # sections
        np_sections = [2, 3]
        self.out = np.split(self.x, np_sections, self.axis)

    def setUp(self):
        self.op_type = "split"
        self.axis_tensor = None
        self.sections_tensor_list = None
        self.num = 0
        self.init_data()
        self.inputs = {'X': self.x}
        self.attrs = {
            'use_mkldnn': True,
            'num': self.num,
            'mkldnn_data_type': "bfloat16",
        }

        if self.axis is not None:
            self.attrs['axis'] = self.axis
        if self.sections is not None:
            self.attrs['sections'] = self.sections
        if self.axis_tensor is not None:
            self.inputs['AxisTensor'] = self.axis_tensor
        if self.sections_tensor_list is not None:
            self.inputs['SectionsTensorList'] = self.sections_tensor_list

        self.outputs = {
            'Out': [(f'out{i}', self.out[i]) for i in range(len(self.out))]
        }

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace(), check_pir_onednn=True)


class TestSplitNumBF16OneDNNOp(TestSplitSectionsBF16OneDNNOp):
    def init_data(self):
        self.x = np.random.random((4, 8, 5, 3)).astype("uint16")
        self.axis = 1
        self.sections = []
        self.num = 4
        indices_or_sections = 4  # indices
        self.out = np.split(self.x, indices_or_sections, self.axis)


class TestSplitNumAxisTensorBF16OneDNNOp(TestSplitSectionsBF16OneDNNOp):
    def init_data(self):
        self.x = np.random.random((4, 5, 6)).astype("uint16")
        self.axis = None
        self.sections = []
        self.num = 3
        indices_or_sections = 3  # indices
        self.axis_tensor = np.array([2]).astype("int32")
        self.out = np.split(self.x, indices_or_sections, 2)


class TestSplitSectionsTensorBF16OneDNNOp(TestSplitSectionsBF16OneDNNOp):
    def init_data(self):
        self.x = np.random.random((4, 5, 6)).astype("uint16")
        self.axis = 1
        self.sections = [2, 1, 2]
        self.sections_tensor_list = []
        for index, ele in enumerate(self.sections):
            self.sections_tensor_list.append(
                ("x" + str(index), np.ones(1).astype('int32') * ele)
            )
        self.sections = [-1, -1, -1]
        indices_or_sections = [2, 3]  # sections
        self.out = np.split(self.x, indices_or_sections, self.axis)


class TestSplitOpUnknownSectionBF16OneDNNOp(TestSplitSectionsBF16OneDNNOp):
    def init_data(self):
        self.x = np.random.random((4, 5, 6)).astype("uint16")
        self.axis = 2
        self.sections = [2, 2, -1]
        indices_or_sections = [2, 4]  # sections
        self.out = np.split(self.x, indices_or_sections, self.axis)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
