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
import paddle.v2.fluid as fluid


class TestCSPFramework(unittest.TestCase):
    def daisy_chain(self):
        n = 10000
        leftmost = fluid.make_channel(dtype=int)
        right = leftmost
        left = leftmost
        with fluid.While(steps=n):
            right = fluid.make_channel(dtype=int)
            with fluid.go():
                fluid.send(left, 1 + fluid.recv(right))
            left = right

        with fluid.go():
            fluid.send(right, 1)
        fluid.Print(fluid.recv(leftmost))


if __name__ == '__main__':
    unittest.main()
