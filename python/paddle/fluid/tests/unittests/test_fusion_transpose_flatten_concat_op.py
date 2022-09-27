#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.core as core


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFusionTransposeFlattenConcationOp(OpTest):

    def setUp(self):
        self.init_test_case()
        self.op_type = "fusion_transpose_flatten_concat"

        ins = []
        flats = []
        for i in range(len(self.shapes)):
            in_shape = self.shapes[i]
            a = np.random.random(in_shape).astype("float32")
            ins.append(("x%d" % i, a))

            b = a.transpose(self.trans_axis)
            flat_shape = (np.prod(b.shape[:self.flatten_axis]),
                          np.prod(b.shape[self.flatten_axis:]))
            c = b.reshape(flat_shape)
            flats.append(c)
        out = np.concatenate(flats, axis=self.concat_axis)

        self.inputs = {'X': ins}
        self.attrs = {
            'trans_axis': list(self.trans_axis),
            'flatten_axis': self.flatten_axis,
            'concat_axis': self.concat_axis
        }
        self.outputs = {'Out': out}

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, 1e-6)

    def init_test_case(self):
        self.shapes = [(3, 4, 17, 17), (3, 8, 7, 7), (3, 12, 5, 5)]
        self.trans_axis = (0, 2, 3, 1)
        self.flatten_axis = 1
        self.concat_axis = 1


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestCase1(TestFusionTransposeFlattenConcationOp):

    def init_test_case(self):
        self.shapes = [(3, 4, 18, 17), (3, 8, 18, 7), (6, 12, 9, 5)]
        self.trans_axis = (0, 2, 3, 1)
        self.flatten_axis = 2
        self.concat_axis = 1


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestCase2(TestFusionTransposeFlattenConcationOp):

    def init_test_case(self):
        self.shapes = [(3, 8, 20, 17), (3, 8, 19, 17), (3, 8, 40, 17)]
        self.trans_axis = (0, 2, 3, 1)
        self.flatten_axis = 2
        self.concat_axis = 0


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestCase3(TestFusionTransposeFlattenConcationOp):

    def init_test_case(self):
        self.shapes = [(3, 8, 20, 17), (3, 8, 19, 17), (3, 8, 40, 17)]
        self.trans_axis = (0, 3, 2, 1)
        self.flatten_axis = 1
        self.concat_axis = 1


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestCase4(TestFusionTransposeFlattenConcationOp):

    def init_test_case(self):
        self.shapes = [(3, 8, 9, 17), (8, 3, 9, 17), (4, 6, 9, 17)]
        self.trans_axis = (0, 2, 1, 3)
        self.flatten_axis = 3
        self.concat_axis = 1


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestCase5(TestFusionTransposeFlattenConcationOp):

    def init_test_case(self):
        self.shapes = [(3, 8, 9, 17, 2), (3, 8, 2, 17, 9), (3, 17, 9, 8, 2)]
        self.trans_axis = (0, 2, 1, 4, 3)
        self.flatten_axis = 1
        self.concat_axis = 1


if __name__ == '__main__':
    unittest.main()
