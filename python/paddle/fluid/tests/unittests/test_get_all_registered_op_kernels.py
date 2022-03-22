#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
from paddle.fluid import core
from paddle import compat as cpt


class TestGetAllRegisteredOpKernels(unittest.TestCase):
    # reshape kernel is in fluid while not in phi
    def test_phi_kernels(self):
        self.assertTrue(core._get_all_register_op_kernels('phi')['sign'])
        with self.assertRaises(KeyError):
            core._get_all_register_op_kernels('phi')['reshape']

    # sign kernel is removed from fluid and added into phi
    def test_fluid_kernels(self):
        self.assertTrue(core._get_all_register_op_kernels('fluid')['reshape'])
        with self.assertRaises(KeyError):
            core._get_all_register_op_kernels('fluid')['sign']

    def test_all_kernels(self):
        self.assertTrue(core._get_all_register_op_kernels('all')['reshape'])
        self.assertTrue(core._get_all_register_op_kernels('all')['sign'])

        self.assertTrue(core._get_all_register_op_kernels()['reshape'])
        self.assertTrue(core._get_all_register_op_kernels()['sign'])


if __name__ == '__main__':
    unittest.main()
