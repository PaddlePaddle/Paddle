import unittest
import paddle.v2.framework.op as op
import paddle.v2.framework.core as core
import paddle.v2.framework.proto.op_proto_pb2 as op_proto_pb2
import paddle.v2.framework.proto.op_desc_pb2 as op_desc_pb2
import paddle.v2.framework.proto.attribute_pb2 as attribute_pb2


class TestGetAllProtos(unittest.TestCase):
    def test_all(self):
        all_protos = op.get_all_op_protos()
        self.assertNotEqual(0, len(all_protos))

        for each in all_protos:
            self.assertTrue(each.IsInitialized())


class TestOpDescCreationMethod(unittest.TestCase):
    def test_plain_input_output(self):
        op_proto = op_proto_pb2.OpProto()
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

        expected = op_desc_pb2.OpDesc()
        expected.type = "test"
        expected.inputs.extend(["a", "b"])
        expected.outputs.append("c")
        self.assertEqual(expected, output)

    def test_multiple_input_plain_output(self):
        op_proto = op_proto_pb2.OpProto()
        op_proto.type = "fc"
        ipt = op_proto.inputs.add()
        ipt.name = "X"
        ipt.comment = ""
        ipt.multiple = True

        ipt = op_proto.inputs.add()
        ipt.name = "W"
        ipt.comment = ""
        ipt.multiple = True

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
        expected1 = op_desc_pb2.OpDesc()
        expected1.inputs.extend(['x', 'w', 'b'])
        expected1.outputs.extend(['y'])
        expected1.type = 'fc'
        # the input_format can be removed after testing
        attr = expected1.attrs.add()
        attr.name = 'input_format'
        attr.type = attribute_pb2.INTS
        attr.ints.extend([0, 1, 2, 3])
        self.assertEqual(expected1, generated1)

        generated2 = method(
            X=['x1', 'x2', 'x3'], b='b', W=['w1', 'w2', 'w3'], Y='y')
        expected2 = op_desc_pb2.OpDesc()
        expected2.inputs.extend(['x1', 'x2', 'x3', 'w1', 'w2', 'w3', 'b'])
        expected2.outputs.extend(['y'])
        expected2.type = 'fc'
        # the input_format can be removed after testing
        attr = expected2.attrs.add()
        attr.name = 'input_format'
        attr.type = attribute_pb2.INTS
        attr.ints.extend([0, 3, 6, 7])
        self.assertEqual(expected2, generated2)

    def test_attrs(self):
        op_proto = op_proto_pb2.OpProto()
        op_proto.type = "test"
        ipt = op_proto.inputs.add()
        ipt.name = 'X'
        ipt.comment = ""

        def __add_attr__(name, type):
            attr = op_proto.attrs.add()
            attr.name = name
            attr.comment = ""
            attr.type = type

        __add_attr__("int_attr", attribute_pb2.INT)
        __add_attr__("float_attr", attribute_pb2.FLOAT)
        __add_attr__("string_attr", attribute_pb2.STRING)
        __add_attr__("ints_attr", attribute_pb2.INTS)
        __add_attr__("floats_attr", attribute_pb2.FLOATS)
        __add_attr__("strings_attr", attribute_pb2.STRINGS)

        op_proto.comment = ""
        self.assertTrue(op_proto.IsInitialized())

        method = op.OpDescCreationMethod(op_proto)

        generated = method(
            X="a",
            int_attr=10,
            float_attr=3.2,
            string_attr="test_str",
            ints_attr=[0, 1, 2, 3, 4],
            floats_attr=[0.2, 3.2, 4.5],
            strings_attr=["a", "b", "c"])

        expected = op_desc_pb2.OpDesc()
        expected.type = "test"
        expected.inputs.extend(['a'])
        attr = expected.attrs.add()
        attr.name = "int_attr"
        attr.type = attribute_pb2.INT
        attr.i = 10

        attr = expected.attrs.add()
        attr.name = "float_attr"
        attr.type = attribute_pb2.FLOAT
        attr.f = 3.2

        attr = expected.attrs.add()
        attr.name = "string_attr"
        attr.type = attribute_pb2.STRING
        attr.s = "test_str"

        attr = expected.attrs.add()
        attr.name = "ints_attr"
        attr.type = attribute_pb2.INTS
        attr.ints.extend([0, 1, 2, 3, 4])

        attr = expected.attrs.add()
        attr.name = 'floats_attr'
        attr.type = attribute_pb2.FLOATS
        attr.floats.extend([0.2, 3.2, 4.5])

        attr = expected.attrs.add()
        attr.name = 'strings_attr'
        attr.type = attribute_pb2.STRINGS
        attr.strings.extend(['a', 'b', 'c'])

        self.assertEqual(expected, generated)

    def test_input_temporary_output(self):
        op_proto = op_proto_pb2.OpProto()
        op_proto.type = "test"
        out = op_proto.outputs.add()
        out.name = "OUT"
        out.comment = ""

        out = op_proto.outputs.add()
        out.name = "TMP"
        out.comment = ""
        out.temporary = True

        out = op_proto.outputs.add()
        out.name = "OUT2"
        out.comment = ""
        op_proto.comment = ""

        method = op.OpDescCreationMethod(op_proto)
        generated = method(OUT="a", OUT2="b")
        desc = op_desc_pb2.OpDesc()
        desc.outputs.extend(["a", core.var_names.temp(), "b"])
        desc.type = "test"
        attr = desc.attrs.add()
        attr.name = "temporary_index"
        attr.type = attribute_pb2.INTS
        attr.ints.append(2)
        self.assertEqual(generated, desc)


class TestOpCreations(unittest.TestCase):
    def test_all(self):
        add_op = op.Operator("add_two", X="a", Y="b", Out="z")
        self.assertIsNotNone(add_op)
        # Invoke C++ DebugString()
        self.assertEqual('Op(add_two), inputs:(a, b), outputs:(z).',
                         str(add_op))


if __name__ == "__main__":
    unittest.main()
