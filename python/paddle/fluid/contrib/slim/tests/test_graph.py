#   copyright (c) 2018 paddlepaddle authors. all rights reserved.
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
from paddle.fluid.framework import IrGraph
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

    data = fluid.layers.data(name='image', shape=[1, 32, 32], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    hidden = data
    for _ in six.moves.xrange(num):
        conv = conv_bn_layer(hidden, 16, 3, 1, 1, act=None, bias_attr=True)
        short = conv_bn_layer(hidden, 16, 1, 1, 0, act=None)
        hidden = fluid.layers.elementwise_add(x=conv, y=short, act='relu')
    fc = fluid.layers.fc(input=hidden, size=10)
    loss = fluid.layers.cross_entropy(input=fc, label=label)
    loss = fluid.layers.mean(loss)
    return loss


class TestGraph(unittest.TestCase):
    def test_graph_functions(self):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            loss = residual_block(2)
            opt = fluid.optimizer.Adam(learning_rate=0.001)
            opt.minimize(loss)
        graph = IrGraph(core.Graph(main.desc), for_test=False)
        marked_nodes = set()
        for op in graph.all_ops():
            if op.name().find('conv2d') > -1:
                marked_nodes.add(op)
        graph.draw('.', 'residual', marked_nodes)
        self.assertFalse(graph.has_circle())
        self.assertEqual(graph.graph_num(), 1)
        nodes = graph.topology_sort()
        self.assertEqual(len(nodes), len(graph.all_ops()))
        nodes_map = graph.build_adjacency_list()
        self.assertEqual(len(nodes_map), len(graph.all_ops()))
        nodes_num = len(graph.all_nodes())
        graph.safe_remove_nodes(marked_nodes)
        self.assertEqual(len(graph.all_nodes()), nodes_num - len(marked_nodes))


if __name__ == '__main__':
    unittest.main()
