# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

from paddle.cinn import Target
from paddle.cinn.framework import Tensor


class TensorTest(unittest.TestCase):
    def test_basic(self):
        target = Target()
        target.arch = Target.X86Arch()
        target.bits = Target.Bit.k64
        target.os = Target.OS.Linux
        tensor = Tensor()
        data = np.random.random([10, 5])
        tensor.from_numpy(data, target)

        np.testing.assert_allclose(tensor.numpy(), data)


if __name__ == "__main__":
    unittest.main()
