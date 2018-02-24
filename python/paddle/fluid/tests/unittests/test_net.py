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

import paddle.fluid.core as core
from paddle.fluid.op import Operator
import unittest


def fc(X, W, Y):
    ret_v = core.Net.create()

    ret_v.append_op(Operator("mul", X="X", Y="W", Out="pre_activation"))
    ret_v.append_op(Operator("sigmoid", X="pre_activation", Out=Y))
    ret_v.complete_add_op(True)
    return ret_v


class TestNet(unittest.TestCase):
    def test_net_all(self):
        net = core.Net.create()
        op1 = Operator("sum", X=["X", "Y"], Out="Out")
        net.append_op(op1)

        net2 = core.Net.create()
        net2.append_op(fc(X="X", W="w", Y="fc.out"))
        net2.complete_add_op(True)
        net.append_op(net2)
        net.complete_add_op(True)

        expected = '''
Op(plain_net), inputs:{all[W, X, Y]}, outputs:{all[Out, fc.out, pre_activation]}.
    Op(sum), inputs:{X[X, Y]}, outputs:{Out[Out]}.
    Op(plain_net), inputs:{all[W, X]}, outputs:{all[fc.out, pre_activation]}.
        Op(plain_net), inputs:{all[W, X]}, outputs:{all[fc.out, pre_activation]}.
            Op(mul), inputs:{X[X], Y[W]}, outputs:{Out[pre_activation]}.
            Op(sigmoid), inputs:{X[pre_activation]}, outputs:{Out[fc.out]}.
'''
        self.assertEqual(expected, "\n" + str(net))


if __name__ == "__main__":
    unittest.main()
