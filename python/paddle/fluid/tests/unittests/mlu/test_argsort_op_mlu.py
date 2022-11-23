#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import numpy as np
import unittest
import sys

sys.path.append("..")
from op_test import OpTest

paddle.enable_static()

SEED = 2022


def gen_test_class(dtype, axis, descending):

    class TestArgsortOp(OpTest):

        def setUp(self):
            np.random.seed(SEED)
            self.set_mlu()
            self.op_type = "argsort"
            self.place = paddle.MLUPlace(0)
            self.init_inputshape()
            if 'int' in dtype:
                self.x = np.random.choice(255, self.size, replace=False)
                self.x = self.x.reshape(self.input_shape).astype(dtype)
            else:
                self.x = np.random.random(self.input_shape).astype(dtype)
            self.inputs = {"X": self.x}
            self.attrs = {"axis": axis, "descending": descending}
            self.get_output()
            self.outputs = {"Out": self.sorted_x, "Indices": self.indices}

        def get_output(self):
            if descending:
                self.indices = np.flip(
                    np.argsort(self.x, kind='heapsort', axis=axis), axis)
                self.sorted_x = np.flip(
                    np.sort(self.x, kind='heapsort', axis=axis), axis)
            else:
                self.indices = np.argsort(self.x, kind='heapsort', axis=axis)
                self.sorted_x = np.sort(self.x, kind='heapsort', axis=axis)

        def test_check_grad(self):
            if dtype in ['float16', 'int8', 'uint8', 'int32']:
                self.__class__.no_need_check_grad = True
            else:
                self.check_grad_with_place(self.place, ["X"], "Out")

        def set_mlu(self):
            self.__class__.use_mlu = True

        def init_inputshape(self):
            self.input_shape = (5, 2, 2, 3, 3)
            self.size = np.prod(self.input_shape)

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def init_direction(self):
            self.descending = False

    cls_name = "{}_{}_{}_TestArgsortOp".format(dtype, axis, descending)
    TestArgsortOp.__name__ = cls_name
    globals()[cls_name] = TestArgsortOp


for dtype in ['float32', 'float16', 'int8', 'uint8', 'int32']:
    for axis in [1, 2, 3, -1]:
        for descending in [False]:
            gen_test_class(dtype, axis, descending)
if __name__ == '__main__':
    unittest.main()
