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

from __future__ import print_function

import unittest
from decorator_helper import prog_scope
import paddle.fluid as fluid
import numpy
from paddle.fluid.dygraph.base import to_variable

class TestMathOpPatches(unittest.TestCase):
    @prog_scope()
    def test_add_scalar(self):
        a = fluid.layers.data(name="a", shape=[1])
        b = a + 10
        ab = fluid.layers.concat(input=[a, b], axis=1)
        c = ab + 10
        d = ab + a
        # e = a + ab
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        a_np = numpy.random.random(size=[10, 1]).astype('float32')
        b_np, c_np, d_np = exe.run(fluid.default_main_program(),
                                   feed={"a": a_np},
                                   fetch_list=[b, c, d])
        self.assertTrue(numpy.allclose(a_np + 10, b_np))
        ab_np = numpy.concatenate([a_np, b_np], axis=1)
        self.assertTrue(numpy.allclose(ab_np + 10, c_np))
        d_expected = ab_np + numpy.concatenate([a_np, a_np], axis=1)
        self.assertTrue(numpy.allclose(d_expected, d_np))

    @prog_scope()
    def test_radd_scalar(self):
        a = fluid.layers.data(name="a", shape=[1])
        b = 10 + a
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        a_np = numpy.random.random(size=[10, 1]).astype('float32')
        b_np = exe.run(fluid.default_main_program(),
                       feed={"a": a_np},
                       fetch_list=[b])
        self.assertTrue(numpy.allclose(a_np + 10, b_np))

    @prog_scope()
    def test_sub_scalar(self):
        a = fluid.layers.data(name="a", shape=[1])
        b = a - 10
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        a_np = numpy.random.random(size=[10, 1]).astype('float32')
        b_np = exe.run(fluid.default_main_program(),
                       feed={"a": a_np},
                       fetch_list=[b])
        self.assertTrue(numpy.allclose(a_np - 10, b_np))

    @prog_scope()
    def test_radd_scalar(self):
        a = fluid.layers.data(name="a", shape=[1])
        b = 10 - a
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        a_np = numpy.random.random(size=[10, 1]).astype('float32')
        b_np = exe.run(fluid.default_main_program(),
                       feed={"a": a_np},
                       fetch_list=[b])
        self.assertTrue(numpy.allclose(10 - a_np, b_np))

    @prog_scope()
    def test_mul_scalar(self):
        a = fluid.layers.data(name="a", shape=[1])
        b = a * 10
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        a_np = numpy.random.random(size=[10, 1]).astype('float32')
        b_np = exe.run(fluid.default_main_program(),
                       feed={"a": a_np},
                       fetch_list=[b])
        self.assertTrue(numpy.allclose(a_np * 10, b_np))

    @prog_scope()
    def test_rmul_scalar(self):
        a = fluid.layers.data(name="a", shape=[1])
        b = 10 * a
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        a_np = numpy.random.random(size=[10, 1]).astype('float32')
        b_np = exe.run(fluid.default_main_program(),
                       feed={"a": a_np},
                       fetch_list=[b])
        self.assertTrue(numpy.allclose(10 * a_np, b_np))

    @prog_scope()
    def test_div_scalar(self):
        a = fluid.layers.data(name="a", shape=[1])
        b = a / 10
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        a_np = numpy.random.random(size=[10, 1]).astype('float32')
        b_np = exe.run(fluid.default_main_program(),
                       feed={"a": a_np},
                       fetch_list=[b])
        self.assertTrue(numpy.allclose(a_np / 10, b_np))

    @prog_scope()
    def test_rdiv_scalar(self):
        a = fluid.layers.data(name="a", shape=[1])
        b = 10 / a
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        a_np = numpy.random.random(size=[10, 1]).astype('float32') + 1e-2

        b_np = exe.run(fluid.default_main_program(),
                       feed={"a": a_np},
                       fetch_list=[b])
        self.assertTrue(numpy.allclose(10 / a_np, b_np))

    @prog_scope()
    def test_div_two_tensor(self):
        a = fluid.layers.data(name="a", shape=[1])
        b = fluid.layers.data(name="b", shape=[1])
        c = a / b
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        a_np = numpy.random.random(size=[10, 1]).astype('float32')
        b_np = numpy.random.random(size=[10, 1]).astype('float32') + 1e-2
        c_np = exe.run(fluid.default_main_program(),
                       feed={"a": a_np,
                             'b': b_np},
                       fetch_list=[c])
        self.assertTrue(numpy.allclose(a_np / b_np, c_np))

    @prog_scope()
    def test_mul_two_tensor(self):
        a = fluid.layers.data(name="a", shape=[1])
        b = fluid.layers.data(name="b", shape=[1])
        c = a * b
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        a_np = numpy.random.random(size=[10, 1]).astype('float32')
        b_np = numpy.random.random(size=[10, 1]).astype('float32')
        c_np = exe.run(fluid.default_main_program(),
                       feed={"a": a_np,
                             'b': b_np},
                       fetch_list=[c])
        self.assertTrue(numpy.allclose(a_np * b_np, c_np))

    @prog_scope()
    def test_add_two_tensor(self):
        a = fluid.layers.data(name="a", shape=[1])
        b = fluid.layers.data(name="b", shape=[1])
        c = a + b
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        a_np = numpy.random.random(size=[10, 1]).astype('float32')
        b_np = numpy.random.random(size=[10, 1]).astype('float32')
        c_np = exe.run(fluid.default_main_program(),
                       feed={"a": a_np,
                             'b': b_np},
                       fetch_list=[c])
        self.assertTrue(numpy.allclose(a_np + b_np, c_np))

    @prog_scope()
    def test_sub_two_tensor(self):
        a = fluid.layers.data(name="a", shape=[1])
        b = fluid.layers.data(name="b", shape=[1])
        c = a - b
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        a_np = numpy.random.random(size=[10, 1]).astype('float32')
        b_np = numpy.random.random(size=[10, 1]).astype('float32')
        c_np = exe.run(fluid.default_main_program(),
                       feed={"a": a_np,
                             'b': b_np},
                       fetch_list=[c])
        self.assertTrue(numpy.allclose(a_np - b_np, c_np))


# class AddOpertion(fluid.dygraph.Layer):
#     def __init__(self, name_scope):
#         super(AddOpertion, self).__init__(name_scope)
#
#     def forward(self, inputs):
#         print(inputs)
#         x = inputs + numpy.array([[5, 6], [7, 8]])
#         return x
#
#
# class TestDygraphMathOpPatches(unittest.TestCase):
#     def test_add_numpy(self):
#         seed = 90
#         epoch_num = 1
#         batch_size = 128
#         with fluid.dygraph.guard():
#             fluid.default_startup_program().random_seed = seed
#             fluid.default_main_program().random_seed = seed
#             inputs = to_variable(numpy.ones([2, 2]))
#             print(inputs)
#             net = AddOpertion("add")
#             # net.train()
#             # net.eval()
#             value = net(inputs)
#             print(value)
#             print(value)
#             print(value)
#             print(value)
#             print(value)
#             print(value)


if __name__ == '__main__':
    unittest.main()
