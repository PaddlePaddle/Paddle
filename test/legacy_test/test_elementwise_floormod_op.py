#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


import unittest

import paddle
from paddle import base


class TestFloorModOp(unittest.TestCase):
    def test_dygraph(self):
        with base.dygraph.guard(base.CPUPlace()):
            # mod by zero
            x = paddle.to_tensor([59], dtype='int32')
            y = paddle.to_tensor([0], dtype='int32')
            try:
                paddle.floor_mod(x, y)
            except Exception as e:
                print("Error: Mod by zero encounter in floor_mod\n")


if __name__ == '__main__':
    unittest.main()
