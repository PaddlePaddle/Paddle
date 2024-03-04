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

from paddle.base import core


class TestBroadcastOpCpu(OpTest):
    def setUp(self):
        self.op_type = "broadcast"
        self.init_dtype()
        input = np.random.random((100, 2)).astype(self.dtype)
        if self.dtype == 'complex64' or self.dtype == 'complex128':
            input = (
                np.random.random((100, 2)) + 1j * np.random.random((100, 2))
            ).astype(self.dtype)
        np_out = input[:]
        self.inputs = {"X": input}
        self.attrs = {"sync_mode": False, "root": 0}
        self.outputs = {"Out": np_out}

    def test_check_output_cpu(self):
        try:
            self.check_output_with_place(place=core.CPUPlace())
        except:
            print("do not support cpu test, skip")

    def init_dtype(self):
        self.dtype = 'float32'


class TestBroadcastOpCpu_complex64(TestBroadcastOpCpu):
    def init_dtype(self):
        self.dtype = 'complex64'


class TestBroadcastOpCpu_complex128(TestBroadcastOpCpu):
    def init_dtype(self):
        self.dtype = 'complex128'


if __name__ == "__main__":
    unittest.main()
