import paddle.v2.fluid as fluid
import paddle.v2.fluid.layers as layers
import op_test
import numpy
import unittest


class TestFetchVar(op_test.OpTest):
    def test_fetch_var(self):
        val = numpy.array([1, 3, 5]).astype(numpy.int32)
        x = layers.create_tensor(dtype="int32", persistable=True, name="x")
        layers.assign(input=val, output=x)
        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_main_program(), feed={}, fetch_list=[])
        fetched_x = fluid.fetch_var("x")
        self.assertTrue(
            numpy.array_equal(fetched_x, val),
            "fetch_x=%s val=%s" % (fetched_x, val))
        self.assertEqual(fetched_x.dtype, val.dtype)


if __name__ == '__main__':
    unittest.main()
