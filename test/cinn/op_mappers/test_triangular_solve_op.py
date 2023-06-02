#!/usr/bin/env python3

# Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

from op_mapper_test import OpMapperTest

import paddle


class TestTriangularSolveOp(OpMapperTest):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([32, 32], "float32"),
            'y': self.random([32, 128], "float32"),
        }

    def set_op_type(self):
        return "triangular_solve"

    def set_op_inputs(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype,
        )
        y = paddle.static.data(
            name='y',
            shape=self.feed_data['y'].shape,
            dtype=self.feed_data['y'].dtype,
        )
        return {'X': [x], 'Y': [y]}

    def set_op_attrs(self):
        return {"upper": True, "transpose": False, "unitriangular": False}

    def set_op_outputs(self):
        return {'Out': [str(self.feed_data['x'].dtype)]}

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestTriangularSolveOpUpper(TestTriangularSolveOp):
    def set_op_attrs(self):
        return {"upper": False, "transpose": False, "unitriangular": False}


class TestTriangularSolveOpTranspose(TestTriangularSolveOp):
    def set_op_attrs(self):
        return {"upper": True, "transpose": True, "unitriangular": False}


class TestTriangularSolveOpUnitriangular(TestTriangularSolveOp):
    def set_op_attrs(self):
        return {"upper": True, "transpose": False, "unitriangular": True}


if __name__ == "__main__":
    unittest.main()
