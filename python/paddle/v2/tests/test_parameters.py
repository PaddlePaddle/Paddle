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


def __rand_param_config__(name, psize=None):
    conf = ParameterConfig()
    conf.name = name
    size = 1
    if psize is None:
        for i in xrange(2):
            dim = random.randint(1, 1000)
            conf.dims.append(dim)
            size *= dim
    else:
        size = psize
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

    def test_init_from_tar(self):
        def get_param(names, size):
            p = parameters.Parameters()
            for k, v in zip(names, size):
                p.__append_config__(__rand_param_config__(k, v))
            for name in p.names():
                param = p.get(name)
                param[:] = numpy.random.uniform(
                    -1.0, 1.0, size=p.get_shape(name))
                p.set(name, param)
            return p

        def get_parames():
            name1 = ['param_0', 'param_1']
            size1 = [128, 256]
            p1 = get_param(name1, size1)
            file1 = cStringIO.StringIO()
            p1.to_tar(file1)
            file1.seek(0)

            name2 = ['param_0', 'param_1', 'param_2']
            size2 = [128, 256, 288]
            p2 = get_param(name2, size2)
            file2 = cStringIO.StringIO()
            p2.to_tar(file2)
            file2.seek(0)
            return p1, file1, p2, file2

        p1, file1, p2, file2 = get_parames()
        p2.init_from_tar(file1)
        for name in p1.names():
            self.assertEqual(p1.get_shape(name), p2.get_shape(name))
            v1 = p1.get(name)
            v2 = p2.get(name)
            self.assertTrue(numpy.isclose(v1, v2).all())

        p1, file1, p2, file2 = get_parames()
        p1.init_from_tar(file2)
        for name in p1.names():
            self.assertEqual(p1.get_shape(name), p2.get_shape(name))
            v1 = p1.get(name)
            v2 = p2.get(name)
            self.assertTrue(numpy.isclose(v1, v2).all())


if __name__ == '__main__':
    unittest.main()
