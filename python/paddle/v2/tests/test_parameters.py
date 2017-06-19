import unittest
import sys

try:
    import py_paddle

    del py_paddle
except ImportError:
    print >> sys.stderr, "It seems swig of Paddle is not installed, this " \
                         "unittest will not be run."
    sys.exit(0)

import paddle.v2.parameters as parameters
import paddle.v2.data_type as data_type
import paddle.v2.layer as layer
from paddle.v2.attr import ParamAttr
from paddle.proto.ParameterConfig_pb2 import ParameterConfig
import random
import cStringIO
import numpy


def __rand_param_config__(name):
    conf = ParameterConfig()
    conf.name = name
    size = 1
    for i in xrange(2):
        dim = random.randint(1, 1000)
        conf.dims.append(dim)
        size *= dim
    conf.size = size
    assert conf.IsInitialized()
    return conf


class TestParameters(unittest.TestCase):
    def test_serialization(self):
        params = parameters.Parameters()
        params.__append_config__(__rand_param_config__("param_0"))
        params.__append_config__(__rand_param_config__("param_1"))

        for name in params.names():
            param = params.get(name)
            param[:] = numpy.random.uniform(
                -1.0, 1.0, size=params.get_shape(name))
            params.set(name, param)

        tmp_file = cStringIO.StringIO()
        params.to_tar(tmp_file)
        tmp_file.seek(0)
        params_dup = parameters.Parameters.from_tar(tmp_file)

        self.assertEqual(params_dup.names(), params.names())

        for name in params.names():
            self.assertEqual(params.get_shape(name), params_dup.get_shape(name))
            p0 = params.get(name)
            p1 = params_dup.get(name)
            self.assertTrue(numpy.isclose(p0, p1).all())

    def test_initializer(self):
        def initializer(name):
            assert name == "fc.w"
            mat = numpy.ones((3, 2), dtype=numpy.float32)
            mat[1, 1] = 2
            return mat

        x = layer.data(name="x", type=data_type.dense_vector(3))
        y = layer.fc(x,
                     size=2,
                     bias_attr=False,
                     param_attr=ParamAttr(
                         name="fc.w", initializer=initializer))
        params = parameters.create(y)
        val = params["fc.w"]
        assert val.shape == (3, 2)
        expected = numpy.array([[1, 1], [1, 2], [1, 1]], numpy.float32)
        assert numpy.logical_and.reduce(numpy.reshape(val == expected, 6))


if __name__ == '__main__':
    unittest.main()
