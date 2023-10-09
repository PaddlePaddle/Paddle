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
import op

from paddle.base.proto import framework_pb2


class TestGetAllProtos(unittest.TestCase):
    def test_all(self):
        all_protos = op.get_all_op_protos()
        self.assertNotEqual(0, len(all_protos))

        for each in all_protos:
            self.assertTrue(each.IsInitialized())


class TestOpDescCreationMethod(unittest.TestCase):
    def test_plain_input_output(self):
        op_proto = framework_pb2.OpProto()
        op_proto.type = "test"
        ipt = op_proto.inputs.add()
        ipt.name = "X"
        ipt.comment = "not matter"

        ipt = op_proto.inputs.add()
        ipt.name = "Y"
        ipt.comment = "not matter"

        opt = op_proto.outputs.add()
        opt.name = "Z"
        opt.comment = "not matter"

        op_proto.comment = "not matter"

        self.assertTrue(op_proto.IsInitialized())

        method = op.OpDescCreationMethod(op_proto)
        output = method(X="a", Y="b", Z="c")
        expected = framework_pb2.OpDesc()
        expected.type = "test"
        ipt_0 = expected.inputs.add()
        ipt_0.parameter = "X"
        ipt_0.arguments.extend(["a"])
        ipt_1 = expected.inputs.add()
        ipt_1.parameter = 'Y'
        ipt_1.arguments.extend(['b'])
        opt = expected.outputs.add()
        opt.parameter = "Z"
        opt.arguments.extend(["c"])

        self.assertEqual(expected, output)

    def test_multiple_input_plain_output(self):
        op_proto = framework_pb2.OpProto()
        op_proto.type = "fc"
        ipt = op_proto.inputs.add()
        ipt.name = "X"
        ipt.comment = ""
        ipt.duplicable = True

        ipt = op_proto.inputs.add()
        ipt.name = "W"
        ipt.comment = ""
        ipt.duplicable = True

        ipt = op_proto.inputs.add()
        ipt.name = "b"
        ipt.comment = ""

        out = op_proto.outputs.add()
        out.name = "Y"
        out.comment = ""

        op_proto.comment = ""
        self.assertTrue(op_proto.IsInitialized())
        method = op.OpDescCreationMethod(op_proto)

        generated1 = method(X="x", W="w", b="b", Y="y")
        expected1 = framework_pb2.OpDesc()
        tmp = expected1.inputs.add()
        tmp.parameter = "X"
        tmp.arguments.extend(['x'])

        tmp = expected1.inputs.add()
        tmp.parameter = 'W'
        tmp.arguments.extend(['w'])

        tmp = expected1.inputs.add()
        tmp.parameter = 'b'
        tmp.arguments.extend(['b'])

        tmp = expected1.outputs.add()
        tmp.parameter = 'Y'
        tmp.arguments.extend(['y'])
        expected1.type = 'fc'
        self.assertEqual(expected1, generated1)

        generated2 = method(
            X=['x1', 'x2', 'x3'], b='b', W=['w1', 'w2', 'w3'], Y='y'
        )
        expected2 = framework_pb2.OpDesc()

        tmp = expected2.inputs.add()
        tmp.parameter = "X"
        tmp.arguments.extend(['x1', 'x2', 'x3'])

        tmp = expected2.inputs.add()
        tmp.parameter = 'W'
        tmp.arguments.extend(['w1', 'w2', 'w3'])

        tmp = expected2.inputs.add()
        tmp.parameter = 'b'
        tmp.arguments.extend(['b'])

        tmp = expected2.outputs.add()
        tmp.parameter = 'Y'
        tmp.arguments.extend(['y'])

        expected2.type = 'fc'
        self.assertEqual(expected2, generated2)

    def test_attrs(self):
        op_proto = framework_pb2.OpProto()
        op_proto.type = "test"
        ipt = op_proto.inputs.add()
        ipt.name = 'X'
        ipt.comment = ""

        def __add_attr__(name, type):
            attr = op_proto.attrs.add()
            attr.name = name
            attr.comment = ""
            attr.type = type

        __add_attr__("int_attr", framework_pb2.INT)
        __add_attr__("float_attr", framework_pb2.FLOAT)
        __add_attr__("float64_attr", framework_pb2.FLOAT64)
        __add_attr__("string_attr", framework_pb2.STRING)
        __add_attr__("ints_attr", framework_pb2.INTS)
        __add_attr__("floats_attr", framework_pb2.FLOATS)
        __add_attr__("strings_attr", framework_pb2.STRINGS)

        op_proto.comment = ""
        self.assertTrue(op_proto.IsInitialized())

        method = op.OpDescCreationMethod(op_proto)

        generated = method(
            X="a",
            int_attr=10,
            float_attr=3.2,
            float64_attr=np.finfo("float64").max,
            string_attr="test_str",
            ints_attr=[0, 1, 2, 3, 4],
            floats_attr=[0.2, 3.2, 4.5],
            strings_attr=["a", "b", "c"],
        )

        expected = framework_pb2.OpDesc()
        expected.type = "test"

        ipt = expected.inputs.add()
        ipt.parameter = "X"
        ipt.arguments.extend(['a'])

        attr = expected.attrs.add()
        attr.name = "int_attr"
        attr.type = framework_pb2.INT
        attr.i = 10

        attr = expected.attrs.add()
        attr.name = "float_attr"
        attr.type = framework_pb2.FLOAT
        attr.f = 3.2

        attr = expected.attrs.add()
        attr.name = "float64_attr"
        attr.type = framework_pb2.FLOAT64
        attr.float64 = np.finfo("float64").max

        attr = expected.attrs.add()
        attr.name = "string_attr"
        attr.type = framework_pb2.STRING
        attr.s = "test_str"

        attr = expected.attrs.add()
        attr.name = "ints_attr"
        attr.type = framework_pb2.INTS
        attr.ints.extend([0, 1, 2, 3, 4])

        attr = expected.attrs.add()
        attr.name = 'floats_attr'
        attr.type = framework_pb2.FLOATS
        attr.floats.extend([0.2, 3.2, 4.5])

        attr = expected.attrs.add()
        attr.name = 'strings_attr'
        attr.type = framework_pb2.STRINGS
        attr.strings.extend(['a', 'b', 'c'])

        self.assertEqual(expected, generated)


class TestOpCreations(unittest.TestCase):
    def test_all(self):
        add_op = op.Operator("sum", X=["a", "b"], Out="z")
        self.assertIsNotNone(add_op)
        # Invoke C++ DebugString()
        self.assertEqual(
            'Op(sum), inputs:{X[a, b]}, outputs:{Out[z]}.', str(add_op)
        )


if __name__ == "__main__":
    unittest.main()
