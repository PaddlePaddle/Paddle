# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


class TestAMPList(unittest.TestCase):
    def test_main(self):
        conv2d = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
        data = paddle.rand([10, 3, 32, 32])
        paddle.set_flags({'FLAGS_low_precision_op_list': 1})
        a = paddle.rand([2, 3])
        b = paddle.rand([2, 3])

        # amp list conv2d, cast
        with paddle.amp.auto_cast():
            conv = conv2d(data)
            c = a + b
        paddle.fluid.dygraph.amp.auto_cast.low_precision_op_list()
        op_list = paddle.fluid.core.get_low_precision_op_list()
        print(conv.dtype)
        if conv.dtype == paddle.float16:
            self.assertTrue('elementwise_add' in op_list)
            self.assertTrue('conv2d' in op_list)
            self.assertTrue(2 == len(op_list))
        else:
            self.assertTrue(0 == len(op_list))


if __name__ == "__main__":
    unittest.main()
