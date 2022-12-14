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

import os
import numpy as np
import unittest
import paddle
import paddle.fluid as fluid
from paddle.fluid.framework import IrGraph
from paddle.fluid import core

paddle.enable_static()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CPU_NUM"] = "1"


def conv_block():
    img = fluid.layers.data(name='image', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu",
    )
    conv_pool_1 = paddle.static.nn.batch_norm(conv_pool_1)
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu",
    )
    prediction = fluid.layers.fc(input=conv_pool_2, size=10, act='softmax')
    loss = paddle.nn.functional.cross_entropy(
        input=prediction, label=label, reduction='none', use_softmax=False
    )
    avg_loss = paddle.mean(loss)
    return [img, label], avg_loss


class TestGraph(unittest.TestCase):
    def graph_apis(self, use_cuda=False, for_ci=True):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main, startup):
                feeds, loss = conv_block()
                opt = fluid.optimizer.Adam(learning_rate=0.001)
                opt.minimize(loss)
        graph = IrGraph(core.Graph(main.desc), for_test=False)
        backup_graph = graph.clone()
        self.assertEqual(len(graph.all_nodes()), len(backup_graph.all_nodes()))
        build_strategy = fluid.BuildStrategy()
        build_strategy.memory_optimize = False
        build_strategy.enable_inplace = False
        origin_binary = fluid.CompiledProgram(graph.graph).with_data_parallel(
            loss_name=loss.name, build_strategy=build_strategy
        )
        backup_binary = fluid.CompiledProgram(
            backup_graph.graph
        ).with_data_parallel(loss_name=loss.name, build_strategy=build_strategy)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup)
        iters = 5
        batch_size = 8
        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=batch_size
        )
        feeder = fluid.DataFeeder(feed_list=feeds, place=place)

        def _train(binary):
            for _ in range(iters):
                data = next(train_reader())
                loss_v = exe.run(
                    binary, feed=feeder.feed(data), fetch_list=[loss.name]
                )
                if not for_ci:
                    print('{}: {}'.format('loss', loss_v))

        _train(origin_binary)
        _train(backup_binary)

        checkponit_dir = "checkpoint_gpu" if use_cuda else "checkpoint_cpu"

        def _set_zero(var_name, scope, place):
            var = scope.find_var(var_name).get_tensor()
            var_array = np.zeros(var._get_dims()).astype("float32")
            var.set(var_array, place)

        sum_before = np.sum(
            np.array(fluid.global_scope().find_var('conv2d_1.w_0').get_tensor())
        )
        fluid.io._save_persistable_nodes(exe, checkponit_dir, graph)
        _set_zero('conv2d_1.w_0', fluid.global_scope(), place)
        set_after = np.sum(
            np.array(fluid.global_scope().find_var('conv2d_1.w_0').get_tensor())
        )
        self.assertEqual(set_after, 0)
        fluid.io._load_persistable_nodes(exe, checkponit_dir, graph)
        sum_after = np.sum(
            np.array(fluid.global_scope().find_var('conv2d_1.w_0').get_tensor())
        )
        self.assertEqual(sum_before, sum_after)

        marked_nodes = set()
        for op in graph.all_op_nodes():
            if op.name().find('conv2d') > -1:
                marked_nodes.add(op)
        if not for_ci:
            graph.draw('.', 'residual', marked_nodes)
            backup_marked_nodes = set()
            for op in backup_graph.all_op_nodes():
                if op.name().find('conv2d') > -1:
                    backup_marked_nodes.add(op)
            backup_graph.draw('./origin', 'backup', backup_marked_nodes)
        self.assertFalse(graph.has_circle())
        self.assertEqual(graph.graph_num(), 1)
        nodes = graph.topology_sort()
        self.assertEqual(len(nodes), len(graph.all_op_nodes()))
        nodes_map = graph.build_adjacency_list()
        self.assertEqual(len(nodes_map), len(graph.all_op_nodes()))
        nodes_num = len(graph.all_nodes())
        graph.safe_remove_nodes(marked_nodes)
        self.assertEqual(len(graph.all_nodes()), nodes_num - len(marked_nodes))

    def test_graph_apis_cpu(self):
        self.graph_apis(use_cuda=False, for_ci=True)

    def test_graph_apis_cuda(self):
        if fluid.core.is_compiled_with_cuda():
            self.graph_apis(use_cuda=True, for_ci=True)


if __name__ == '__main__':
    unittest.main()
