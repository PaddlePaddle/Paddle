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

import paddle.fluid as fluid
import graphviz
import unittest
import numpy


def progdesc_to_diagraph(prog):
    assert isinstance(prog, fluid.Program)
    graph = graphviz.Digraph()

    def append_block(g, block):
        assert isinstance(g, graphviz.Digraph)
        var_node_dict = dict()
        for var_id, var_name in enumerate(block.vars):
            name = 'var_{0}'.format(var_id)
            g.node(name=name, label=var_name)
            var_node_dict[var_name] = name

        for op_id, op in enumerate(block.ops):
            name = 'op_{0}'.format(op_id)
            g.node(name=name, label=op.type, shape='box')
            assert isinstance(op, fluid.Operator)
            for out_name in op.output_arg_names:
                g.edge(head_name=var_node_dict[out_name], tail_name=name)
            for in_name in op.input_arg_names:
                g.edge(head_name=name, tail_name=var_node_dict[in_name])

    append_block(graph, prog.global_block())
    return graph


def simple_fc():
    img = fluid.layers.data(name='img', shape=[784])
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    hidden = img
    for i in xrange(1):
        hidden = fluid.layers.fc(input=img, size=200, act='sigmoid')
        # hidden = fluid.layers.dropout(hidden, dropout_prob=0.1, seed=1)
        hidden = fluid.layers.batch_norm(
            hidden,
            moving_mean_name='moving_mean.{0}'.format(i),
            moving_variance_name='moving_var.{0}'.format(i))
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    loss = fluid.layers.mean(loss)
    adam = fluid.optimizer.Adam()
    adam.minimize(loss)
    return loss


class FCDataRandom(object):
    def __init__(self, batch_size=64):
        self.random_state = numpy.random.RandomState(seed=1)
        self.batch_size = batch_size

    def next(self):
        return {
            'img': self.random_state.uniform(
                low=-1, high=1, size=(self.batch_size, 784)).astype('float32'),
            'label': self.random_state.uniform(
                low=0.0, high=10.0, size=(self.batch_size, 1)).astype('int64')
        }


def create_unittest(network_func, data_random):
    class __cls__(unittest.TestCase):
        def test_main(self):
            startup = fluid.Program()
            startup.random_seed = 1
            main = fluid.Program()
            with fluid.program_guard(main, startup):
                loss = network_func()

            mem_opt_main = main.clone()
            fluid.memory_optimize(mem_opt_main)

            graph = progdesc_to_diagraph(mem_opt_main)
            print graph

            place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            exe.run(startup)
            data = data_random()
            pe = fluid.ParallelExecutor(
                use_cuda=True, loss_name=loss.name, main_program=main)
            for i in xrange(1000):
                pe.run(fetch_list=[], feed=next(data))
            loss_value = numpy.array(
                pe.run(fetch_list=[loss.name], feed=next(data))[0])
            del pe
            data = data_random()

            exe.run(startup)
            pe = fluid.ParallelExecutor(
                use_cuda=True, loss_name=loss.name, main_program=mem_opt_main)
            for i in xrange(1000):
                pe.run(fetch_list=[], feed=next(data))

            loss_value_mem_opted = numpy.array(
                pe.run(fetch_list=[loss.name], feed=next(data))[0])
            self.assertAlmostEqual(loss_value[0], loss_value_mem_opted[0])

    return __cls__


TestSimpleFC = create_unittest(simple_fc, FCDataRandom)

if __name__ == '__main__':
    unittest.main()
