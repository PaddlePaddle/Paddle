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

import unittest
import numpy as np
from op_test import OpTest
import paddle.nn.functional as F
import paddle.fluid as fluid
import paddle.fluid.dygraph as dg
import paddle.fluid.core as core


class TestDiagEmbedOp(OpTest):

    def setUp(self):
        self.op_type = "diag_embed"
        self.python_api = F.diag_embed
        self.init_config()
        self.outputs = {'Out': self.target}

    def test_check_output(self):
        self.check_output(check_eager=True)

    def init_config(self):
        self.case = np.random.randn(2, 3).astype('float32')
        self.inputs = {'Input': self.case}
        self.attrs = {'offset': 0, 'dim1': -2, 'dim2': -1}
        self.target = np.stack([np.diag(r, 0) for r in self.inputs['Input']], 0)


class TestDiagEmbedOpCase1(TestDiagEmbedOp):

    def init_config(self):
        self.case = np.random.randn(2, 3).astype('float32')
        self.inputs = {'Input': self.case}
        self.attrs = {'offset': -1, 'dim1': 0, 'dim2': 2}
        self.target = np.stack([np.diag(r, -1) for r in self.inputs['Input']],
                               1)


class TestDiagEmbedAPICase(unittest.TestCase):

    def test_case1(self):
        diag_embed = np.random.randn(2, 3, 4).astype('float32')
        data1 = fluid.data(name='data1', shape=[2, 3, 4], dtype='float32')
        out1 = F.diag_embed(data1)
        out2 = F.diag_embed(data1, offset=1, dim1=-2, dim2=3)

        place = core.CPUPlace()
        exe = fluid.Executor(place)
        results = exe.run(fluid.default_main_program(),
                          feed={"data1": diag_embed},
                          fetch_list=[out1, out2],
                          return_numpy=True)
        target1 = np.stack(
            [np.stack([np.diag(s, 0) for s in r], 0) for r in diag_embed], 0)
        target2 = np.stack(
            [np.stack([np.diag(s, 1) for s in r], 0) for r in diag_embed], 0)
        np.testing.assert_allclose(results[0], target1, rtol=1e-05)
        np.testing.assert_allclose(results[1], target2, rtol=1e-05)


if __name__ == "__main__":
    unittest.main()
