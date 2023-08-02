#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

paddle.enable_static()
np.random.seed(10)


class XPUTestExpandOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = "expand"
        self.use_dynamic_create_class = False

    # Situation 1: expand_times is a list(without tensor)
    class TestXPUExpandOpRank1(XPUOpTest):
        def setUp(self):
            self.op_type = "expand"
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.init_data()

            self.inputs = {
                'X': np.random.random(self.ori_shape).astype(self.dtype)
            }
            self.attrs = {'expand_times': self.expand_times}
            self.outputs = {'Out': np.tile(self.inputs['X'], self.expand_times)}

        def init_data(self):
            self.ori_shape = [100]
            self.expand_times = [2]

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ['X'], 'Out')

    class TestXPUExpandOpRank1_Cornet(TestXPUExpandOpRank1):
        def init_data(self):
            self.ori_shape = [120]
            self.expand_times = [2]

    class TestXPUExpandOpRank2(TestXPUExpandOpRank1):
        def init_data(self):
            self.ori_shape = [12, 14]
            self.expand_times = [2, 3]

    class TestXPUExpandOpRank3(TestXPUExpandOpRank1):
        def init_data(self):
            self.ori_shape = [2, 4, 15]
            self.expand_times = [2, 1, 3]

    class TestXPUExpandOpRank3_Corner(TestXPUExpandOpRank1):
        def init_data(self):
            self.ori_shape = [2, 10, 5]
            self.expand_times = [1, 1, 1]

    class TestXPUExpandOpRank4(TestXPUExpandOpRank1):
        def init_data(self):
            self.ori_shape = [2, 4, 15, 7]
            self.expand_times = [2, 1, 3, 2]

    # Situation 2: expand_times is a list(with tensor)
    class TestXPUExpandOpRank1_tensor_attr(TestXPUExpandOpRank1):
        def setUp(self):
            self.op_type = "expand"
            # self.dtype = self.in_type
            self.dtype = np.float32
            self.place = paddle.XPUPlace(0)
            self.init_data()

            expand_times_tensor = []
            for index, ele in enumerate(self.expand_times):
                expand_times_tensor.append(
                    ("x" + str(index), np.ones(1).astype('int32') * ele)
                )

            self.inputs = {
                'X': np.random.random(self.ori_shape).astype(self.dtype),
                'expand_times_tensor': expand_times_tensor,
            }
            self.attrs = {"expand_times": self.infer_expand_times}
            self.outputs = {'Out': np.tile(self.inputs['X'], self.expand_times)}

        def init_data(self):
            self.ori_shape = [100]
            self.expand_times = [2]
            self.infer_expand_times = [-1]

    class TestXPUExpandOpRank2_tensor_attr(TestXPUExpandOpRank1_tensor_attr):
        def init_data(self):
            self.ori_shape = [12, 14]
            self.expand_times = [2, 3]
            self.infer_expand_times = [-1, 3]

    class TestXPUExpandOpRank2_Corner_tensor_attr(
        TestXPUExpandOpRank1_tensor_attr
    ):
        def init_data(self):
            self.ori_shape = [12, 14]
            self.expand_times = [1, 1]
            self.infer_expand_times = [1, -1]

    # Situation 3: expand_times is a tensor
    class TestXPUExpandOpRank1_tensor(TestXPUExpandOpRank1):
        def setUp(self):
            self.op_type = "expand"
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.init_data()

            self.inputs = {
                'X': np.random.random(self.ori_shape).astype(self.dtype),
                'ExpandTimes': np.array(self.expand_times).astype("int32"),
            }
            self.attrs = {}
            self.outputs = {'Out': np.tile(self.inputs['X'], self.expand_times)}

        def init_data(self):
            self.ori_shape = [100]
            self.expand_times = [2]

    class TestXPUExpandOpRank2_tensor(TestXPUExpandOpRank1_tensor):
        def init_data(self):
            self.ori_shape = [12, 14]
            self.expand_times = [2, 3]


support_types = get_xpu_op_support_types('expand')
for stype in support_types:
    create_test_class(globals(), XPUTestExpandOp, stype)

if __name__ == "__main__":
    unittest.main()
