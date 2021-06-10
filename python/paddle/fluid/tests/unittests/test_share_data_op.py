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
from paddle.fluid import core
from paddle.fluid.op import Operator


class TestShareDataOp(OpTest):
    def setUp(self):
        self.op_type = "share_data"
        input = np.random.rand(2, 3, 5).astype("float32")
        self.inputs = {'Input': input}
        self.outputs = {'Out': input}

    def test_check_output(self):
        self.check_output()


class TestShareDataOpOnDifferentPlaces(unittest.TestCase):
    def get_places(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        return places

    def check_with_place(self, place):
        scope = core.Scope()
        np_array = np.random.rand(2, 3, 5).astype("float32")

        # initialize input and output variable
        x = scope.var('Input').get_tensor()
        x.set(np_array, place)
        out = scope.var("Out").get_tensor()

        op = Operator("share_data", Input="Input", Out="Out")
        op.run(scope, place)
        self.assertTrue(np.allclose(np_array, out))

    def test_check_output(self):
        for place in self.get_places():
            self.check_with_place(place)


if __name__ == '__main__':
    unittest.main()
