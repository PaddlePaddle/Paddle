import unittest
import paddle.v2.framework.create_op_creation_methods as creation
import paddle.v2.framework.core as core
import paddle.v2.framework.proto.op_proto_pb2 as op_proto_pb2
import paddle.v2.framework.proto.op_desc_pb2 as op_desc_pb2
import paddle.v2.framework.proto.attr_type_pb2 as attr_type_pb2


class TestGetAllProtos(unittest.TestCase):
    def test_all(self):
        all_protos = creation.get_all_op_protos()
        self.assertNotEqual(0, len(all_protos))

        for each in all_protos:
            self.assertTrue(each.IsInitialized())


class TestOpDescCreationMethod(unittest.TestCase):
    def test_plain_input_output(self):
        op = op_proto_pb2.OpProto()
        op.type = "test"
        ipt = op.inputs.add()
        ipt.name = "X"
        ipt.comment = "not matter"

        ipt = op.inputs.add()
        ipt.name = "Y"
        ipt.comment = "not matter"

        opt = op.outputs.add()
        opt.name = "Z"
        opt.comment = "not matter"

        op.comment = "not matter"

        self.assertTrue(op.IsInitialized())

        method = creation.OpDescCreationMethod(op)
        output = method(X="a", Y="b", Z="c")

        expected = op_desc_pb2.OpDesc()
        expected.type = "test"
        expected.inputs.extend(["a", "b"])
        expected.outputs.append("c")
        self.assertEqual(expected, output)

    def test_multiple_input_plain_output(self):
        op = op_proto_pb2.OpProto()
        op.type = "fc"
        ipt = op.inputs.add()
        ipt.name = "X"
        ipt.comment = ""
        ipt.multiple = True

        ipt = op.inputs.add()
        ipt.name = "W"
        ipt.comment = ""
        ipt.multiple = True

        ipt = op.inputs.add()
        ipt.name = "b"
        ipt.comment = ""

        out = op.outputs.add()
        out.name = "Y"
        out.comment = ""

        op.comment = ""
        self.assertTrue(op.IsInitialized())
        method = creation.OpDescCreationMethod(op)

        generated1 = method(X="x", W="w", b="b", Y="y")
        expected1 = op_desc_pb2.OpDesc()
        expected1.inputs.extend(['x', 'w', 'b'])
        expected1.outputs.extend(['y'])
        expected1.type = 'fc'
        attr = expected1.attrs.add()
        attr.name = 'input_format'
        attr.type = attr_type_pb2.INTS
        attr.ints.extend([0, 1, 2, 3])
        self.assertEqual(expected1, generated1)

        generated2 = method(
            X=['x1', 'x2', 'x3'], b='b', W=['w1', 'w2', 'w3'], Y='y')
        expected2 = op_desc_pb2.OpDesc()
        expected2.inputs.extend(['x1', 'x2', 'x3', 'w1', 'w2', 'w3', 'b'])
        expected2.outputs.extend(['y'])
        expected2.type = 'fc'
        attr = expected2.attrs.add()
        attr.name = 'input_format'
        attr.type = attr_type_pb2.INTS
        attr.ints.extend([0, 3, 6, 7])
        self.assertEqual(expected2, generated2)

    def test_attrs(self):
        op = op_proto_pb2.OpProto()
        op.type = "test"
        ipt = op.inputs.add()
        ipt.name = 'X'
        ipt.comment = ""

        def __add_attr__(name, type):
            attr = op.attrs.add()
            attr.name = name
            attr.comment = ""
            attr.type = type

        __add_attr__("int_attr", attr_type_pb2.INT)
        __add_attr__("float_attr", attr_type_pb2.FLOAT)
        __add_attr__("string_attr", attr_type_pb2.STRING)
        __add_attr__("ints_attr", attr_type_pb2.INTS)
        __add_attr__("floats_attr", attr_type_pb2.FLOATS)
        __add_attr__("strings_attr", attr_type_pb2.STRINGS)

        op.comment = ""
        self.assertTrue(op.IsInitialized())

        method = creation.OpDescCreationMethod(op)

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
        attr.type = attr_type_pb2.INT
        attr.i = 10

        attr = expected.attrs.add()
        attr.name = "float_attr"
        attr.type = attr_type_pb2.FLOAT
        attr.f = 3.2

        attr = expected.attrs.add()
        attr.name = "string_attr"
        attr.type = attr_type_pb2.STRING
        attr.s = "test_str"

        attr = expected.attrs.add()
        attr.name = "ints_attr"
        attr.type = attr_type_pb2.INTS
        attr.ints.extend([0, 1, 2, 3, 4])

        attr = expected.attrs.add()
        attr.name = 'floats_attr'
        attr.type = attr_type_pb2.FLOATS
        attr.floats.extend([0.2, 3.2, 4.5])

        attr = expected.attrs.add()
        attr.name = 'strings_attr'
        attr.type = attr_type_pb2.STRINGS
        attr.strings.extend(['a', 'b', 'c'])

        self.assertEqual(expected, generated)

    def test_input_temporary_output(self):
        op = op_proto_pb2.OpProto()
        op.type = "test"
        out = op.outputs.add()
        out.name = "OUT"
        out.comment = ""

        out = op.outputs.add()
        out.name = "TMP"
        out.comment = ""
        out.temporary = True

        out = op.outputs.add()
        out.name = "OUT2"
        out.comment = ""
        op.comment = ""

        method = creation.OpDescCreationMethod(op)
        generated = method(OUT="a", OUT2="b")
        desc = op_desc_pb2.OpDesc()
        desc.outputs.extend(["a", core.var_names.temp(), "b"])
        desc.type = "test"
        attr = desc.attrs.add()
        attr.name = "temporary_index"
        attr.type = attr_type_pb2.INTS
        attr.ints.append(2)
        self.assertEqual(generated, desc)


class TestOpCreationDocStr(unittest.TestCase):
    def test_all(self):
        op = op_proto_pb2.OpProto()
        op.type = "test"
        op.comment = """Test Op.

This op is used for unit test, not a real op.
"""
        a = op.inputs.add()
        a.name = "a"
        a.comment = "Input a for test op"
        a.multiple = True

        b = op.inputs.add()
        b.name = "b"
        b.comment = "Input b for test op"
        self.assertTrue(op.IsInitialized())

        o1 = op.outputs.add()
        o1.name = "output"
        o1.comment = "The output of test op"

        o2 = op.outputs.add()
        o2.name = "temp output"
        o2.comment = "The temporary output of test op"
        o2.temporary = True

        test_str = op.attrs.add()
        test_str.name = "str_attr"
        test_str.type = attr_type_pb2.STRING
        test_str.comment = "A string attribute for test op"

        actual = creation.get_docstring_from_op_proto(op)
        expected_docstring = '''Test Op.

This op is used for unit test, not a real op.

:param a: Input a for test op
:type a: list | basestr
:param b: Input b for test op
:type b: basestr
:param output: The output of test op
:type output: basestr
:param temp output: This is a temporary variable. It does not have to set by user. The temporary output of test op
:type temp output: basestr
:param str_attr: A string attribute for test op
:type str_attr: basestr
'''
        self.assertEqual(expected_docstring, actual)


class TestOpCreations(unittest.TestCase):
    def test_all(self):
        add_op = creation.op_creations.add_two(X="a", Y="b", Out="z")
        self.assertIsNotNone(add_op)
        # Invoke C++ DebugString()
        self.assertEqual('Op(add_two), inputs:(a, b), outputs:(z).',
                         str(add_op))


if __name__ == "__main__":
    unittest.main()
