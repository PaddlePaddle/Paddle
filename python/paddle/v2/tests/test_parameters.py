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


if __name__ == '__main__':
    unittest.main()
