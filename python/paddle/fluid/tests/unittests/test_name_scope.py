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
import paddle.fluid as fluid


class TestNameScope(unittest.TestCase):

    def test_name_scope(self):
        with fluid.name_scope("s1"):
            a = fluid.layers.data(name='data', shape=[1], dtype='int32')
            b = a + 1
            with fluid.name_scope("s2"):
                c = b * 1
            with fluid.name_scope("s3"):
                d = c / 1
        with fluid.name_scope("s1"):
            f = fluid.layers.pow(d, 2.0)
        with fluid.name_scope("s4"):
            g = f - 1

        for op in fluid.default_main_program().block(0).ops:
            if op.type == 'elementwise_add':
                self.assertEqual(op.desc.attr("op_namescope"), '/s1/')
            elif op.type == 'elementwise_mul':
                self.assertEqual(op.desc.attr("op_namescope"), '/s1/s2/')
            elif op.type == 'elementwise_div':
                self.assertEqual(op.desc.attr("op_namescope"), '/s1/s3/')
            elif op.type == 'elementwise_sub':
                self.assertEqual(op.desc.attr("op_namescope"), '/s4/')
            elif op.type == 'pow':
                self.assertEqual(op.desc.attr("op_namescope"), '/s1_1/')


if __name__ == "__main__":
    unittest.main()
