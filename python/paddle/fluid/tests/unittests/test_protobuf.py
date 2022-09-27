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

import paddle.fluid.proto.framework_pb2 as framework_pb2
import unittest


class TestFrameworkProto(unittest.TestCase):

    def test_all(self):
        op_proto = framework_pb2.OpProto()
        ipt0 = op_proto.inputs.add()
        ipt0.name = "a"
        ipt0.comment = "the input of cosine op"
        ipt1 = op_proto.inputs.add()
        ipt1.name = "b"
        ipt1.comment = "the other input of cosine op"
        opt = op_proto.outputs.add()
        opt.name = "output"
        opt.comment = "the output of cosine op"
        op_proto.comment = "cosine op, output = scale*cos(a, b)"
        attr = op_proto.attrs.add()
        attr.name = "scale"
        attr.comment = "scale of cosine op"
        attr.type = framework_pb2.FLOAT
        op_proto.type = "cos"
        self.assertTrue(op_proto.IsInitialized())


if __name__ == "__main__":
    unittest.main()
