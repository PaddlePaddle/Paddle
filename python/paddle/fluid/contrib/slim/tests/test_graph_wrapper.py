#   copyright (c) 2019 paddlepaddle authors. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.

from __future__ import print_function
import unittest
import paddle.fluid as fluid
import six
import numpy as np
from paddle.fluid.contrib.slim.graph import GraphWrapper
from paddle.fluid import core


def residual_block(num):
    def conv_bn_layer(input,
                      ch_out,
                      filter_size,
                      stride,
                      padding,
                      act='relu',
                      bias_attr=False):
        tmp = fluid.layers.conv2d(
            input=input,
            filter_size=filter_size,
            num_filters=ch_out,
            stride=stride,
            padding=padding,
            act=None,
            bias_attr=bias_attr)
        return fluid.layers.batch_norm(input=tmp, act=act)

    data = fluid.layers.data(name='image', shape=[1, 8, 8], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    data.stop_gradinet = False
    hidden = data
    for _ in six.moves.xrange(num):
        conv = conv_bn_layer(hidden, 16, 3, 1, 1, act=None, bias_attr=True)
        short = conv_bn_layer(hidden, 16, 1, 1, 0, act=None)
        hidden = fluid.layers.elementwise_add(x=conv, y=short, act='relu')
    fc = fluid.layers.fc(input=hidden, size=10)

    loss = fluid.layers.cross_entropy(input=fc, label=label)
    loss = fluid.layers.mean(loss)
    return data, label, loss


class TestGraphWrapper(unittest.TestCase):
    def build_program(self):
        place = fluid.CPUPlace()
        if fluid.core.is_compiled_with_cuda():
            place = fluid.CUDAPlace(0)
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            image, label, self.loss = residual_block(2)
            eval_program = main.clone()
            opt = fluid.optimizer.SGD(learning_rate=0.001)
            opt.minimize(self.loss)
        self.scope = core.Scope()
        exe = fluid.Executor(place)
        exe.run(startup, scope=self.scope)
        self.eval_graph = GraphWrapper(
            program=eval_program,
            in_nodes={'image': image.name,
                      'label': label.name},
            out_nodes={'loss': self.loss.name})
        self.train_graph = GraphWrapper(
            program=main,
            in_nodes={'image': image.name,
                      'label': label.name},
            out_nodes={'loss': self.loss.name})

    def test_all_parameters(self):
        self.build_program()
        self.assertEquals(len(self.train_graph.all_parameters()), 24)

    def test_all_vars(self):
        self.build_program()
        # self.assertEquals(len(self.train_graph.vars()), 90)
        # activation inplace has been disabled in python side
        # which may produce more variable in program_desc
        # update 90 => 94
        self.assertEquals(len(self.train_graph.vars()), 94)

    def test_numel_params(self):
        self.build_program()
        self.assertEquals(self.train_graph.numel_params(), 13258)

    def test_compile(self):
        self.build_program()
        place = fluid.CPUPlace()
        if fluid.core.is_compiled_with_cuda():
            place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        self.train_graph.compile()
        exe.run(self.train_graph.compiled_graph,
                scope=self.scope,
                feed={
                    'image':
                    np.random.randint(0, 40, [16, 1, 8, 8]).astype('float32'),
                    'label': np.random.randint(0, 10, [16, 1]).astype('int64')
                })

    def test_pre_and_next_ops(self):
        self.build_program()
        for op in self.train_graph.ops():
            for next_op in self.train_graph.next_ops(op):
                self.assertTrue(op in self.train_graph.pre_ops(next_op))

    def test_get_optimize_graph(self):
        self.build_program()
        place = fluid.CPUPlace()
        if fluid.core.is_compiled_with_cuda():
            place = fluid.CUDAPlace(0)
        opt = fluid.optimizer.SGD(learning_rate=0.001)
        train_graph = self.eval_graph.get_optimize_graph(
            opt, place, self.scope, no_grad_var_names=['image'])
        self.assertEquals(len(self.train_graph.ops()), len(train_graph.ops()))
        exe = fluid.Executor(place)
        train_graph.compile()
        image = np.random.randint(0, 225, [16, 1, 8, 8]).astype('float32')
        label = np.random.randint(0, 10, [16, 1]).astype('int64')
        exe.run(train_graph.compiled_graph,
                scope=self.scope,
                feed={'image': image,
                      'label': label})

    def test_flops(self):
        self.build_program()
        self.assertEquals(self.train_graph.flops(), 354624)


if __name__ == '__main__':
    unittest.main()
