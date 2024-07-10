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

import paddle
from paddle import base


class TestSigmoidCrossEntropyWithLogitsOpError(unittest.TestCase):
    def test_errors(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):

            def test_Variable():
                # the input of sigmoid_cross_entropy_with_logits must be Variable.
                x1 = base.create_lod_tensor(
                    np.array([-1, 3, 5, 5]),
                    [[1, 1, 1, 1]],
                    base.CPUPlace(),
                )
                lab1 = base.create_lod_tensor(
                    np.array([-1, 3, 5, 5]),
                    [[1, 1, 1, 1]],
                    base.CPUPlace(),
                )
                paddle.nn.functional.binary_cross_entropy_with_logits(x1, lab1)

            self.assertRaises(TypeError, test_Variable)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
