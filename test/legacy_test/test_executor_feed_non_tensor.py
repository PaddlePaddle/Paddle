#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import base


class TestAsLodTensor(unittest.TestCase):
    def test_as_lodtensor_int32(self):
        cpu = base.CPUPlace()
        tensor = base.executor._as_lodtensor(1.0, cpu, paddle.int32)
        self.assertEqual(tensor._dtype(), base.core.VarDesc.VarType.INT32)

    def test_as_lodtensor_fp64(self):
        cpu = base.CPUPlace()
        tensor = base.executor._as_lodtensor(1, cpu, paddle.float64)
        self.assertEqual(tensor._dtype(), base.core.VarDesc.VarType.FP64)

    def test_as_lodtensor_assertion_error(self):
        cpu = base.CPUPlace()
        self.assertRaises(AssertionError, base.executor._as_lodtensor, 1, cpu)

    def test_as_lodtensor_type_error(self):
        cpu = base.CPUPlace()
        self.assertRaises(
            TypeError,
            base.executor._as_lodtensor,
            {"a": 1},
            cpu,
            base.core.VarDesc.VarType.INT32,
        )

    def test_as_lodtensor_list(self):
        cpu = base.CPUPlace()
        tensor = base.executor._as_lodtensor([1, 2], cpu, paddle.float64)
        self.assertEqual(tensor._dtype(), base.core.VarDesc.VarType.FP64)

    def test_as_lodtensor_tuple(self):
        cpu = base.CPUPlace()
        tensor = base.executor._as_lodtensor((1, 2), cpu, paddle.float64)
        self.assertEqual(tensor._dtype(), base.core.VarDesc.VarType.FP64)

    def test_as_lodtensor_nested_list(self):
        cpu = base.CPUPlace()
        self.assertRaises(
            TypeError,
            base.executor._as_lodtensor,
            [{1.2, 1.2}, {1, 2}],
            cpu,
            base.core.VarDesc.VarType.INT32,
        )


if __name__ == '__main__':
    unittest.main()
