# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
from paddle.fluid.contrib.slim import SensitivePruneStrategy
from paddle.fluid.contrib.slim import StructurePruner
from paddle.fluid.contrib.slim import Context
from paddle.fluid.contrib.slim import ImitationGraph
from paddle.fluid.contrib.slim import get_executor
import unittest


class TestSensitivePruneStrategy(unittest.TestCase):
    def def_graph(self):
        program = fluid.Program()
        startup_program = fluid.Program()
        filter_size = 3
        with fluid.program_guard(program, startup_program):
            with fluid.unique_name.guard():
                img = fluid.layers.ones(shape=[1, 3, 4, 4], dtype="float32")
                label = fluid.layers.ones(shape=[1, 1], dtype="int64")
                conv_0 = fluid.layers.conv2d(img, 4, filter_size, 1, 1)
                conv_1 = fluid.layers.conv2d(img, 3, filter_size, 1, 1)
                concat = fluid.layers.concat([conv_0, conv_1], axis=1)
                conv_2 = fluid.layers.conv2d(concat, 4, filter_size, 1, 1)
                conv_3 = fluid.layers.conv2d(conv_2, 4, filter_size, 1, 1)
                conv_4 = fluid.layers.conv2d(conv_3, 4, filter_size, 1, 1)
                conv_5 = fluid.layers.conv2d(conv_2, 4, filter_size, 1, 1)
                add_0 = conv_4 + conv_5
                conv_6 = fluid.layers.conv2d(add_0, 4, 2)
                fc_0 = fluid.layers.fc(conv_6, size=2)
                loss = fluid.layers.softmax_with_cross_entropy(fc_0, label)
        graph = ImitationGraph(program)
        place = fluid.CPUPlace()
        p_exe = fluid.Executor(place)
        scope = fluid.core.Scope()
        p_exe.run(startup_program, scope=scope)
        exe = get_executor(graph, place)
        return graph, scope, exe, place, loss

    def test_compute_sensitivities(self):
        graph, scope, exe, place, loss = self.def_graph()
        context = Context(exe, graph, scope, place)
        context.put('sensitivity_metric', loss)
        ratios = {}
        pruning_axis = {'*': 0}
        criterions = {'*': 'l1_norm'}
        pruner = StructurePruner(ratios, pruning_axis, criterions)
        strategy = SensitivePruneStrategy(pruner=pruner)
        strategy._compute_sensitivities(context)

    def test_prune_parameter(self):
        graph, scope, exe, place, loss = self.def_graph()
        pruning_axis = {'*': 0}
        criterions = {'*': 'l1_norm'}
        pruner = StructurePruner(None, pruning_axis, criterions)
        strategy = SensitivePruneStrategy(pruner=pruner)
        param = graph.get_var('conv2d_1.w_0')
        param_t = scope.find_var(param.name).get_tensor()
        orig_shape = list(np.array(param_t).shape)
        ratio = 0.3
        strategy._prune_parameter(scope, param, ratio, place)
        pruned_shape = list(np.array(param_t).shape)
        orig_shape[0] -= round(ratio * orig_shape[0])
        for v0, v1 in zip(pruned_shape, orig_shape):
            self.assertEquals(v0, v1)

    def test_prune_parameters_lazy(self):
        graph, scope, exe, place, loss = self.def_graph()
        pruning_axis = {'*': 0}
        criterions = {'*': 'l1_norm'}
        pruner = StructurePruner(None, pruning_axis, criterions)
        strategy = SensitivePruneStrategy(pruner=pruner)

        # pruning conv_1
        param = graph.get_var('conv2d_1.w_0')
        ratio = 0.3
        strategy._prune_parameters(
            graph, scope, [param], [0.3], place, lazy=True)
        self.assertTrue(len(strategy.backup.keys()) == 3)
        self.assertTrue('conv2d_1.w_0' in strategy.backup.keys())
        self.assertTrue('conv2d_1.b_0' in strategy.backup.keys())
        self.assertTrue('conv2d_2.w_0' in strategy.backup.keys())

        # pruning conv_2
        strategy.backup = {}
        param = graph.get_var('conv2d_2.w_0')
        ratio = 0.3
        strategy._prune_parameters(
            graph, scope, [param], [0.3], place, lazy=True)
        self.assertTrue(len(strategy.backup.keys()) == 4)
        self.assertTrue('conv2d_2.w_0' in strategy.backup.keys())
        self.assertTrue('conv2d_2.b_0' in strategy.backup.keys())
        self.assertTrue('conv2d_3.w_0' in strategy.backup.keys())
        self.assertTrue('conv2d_5.w_0' in strategy.backup.keys())

        # pruning conv_3
        strategy.backup = {}
        param = graph.get_var('conv2d_3.w_0')
        ratio = 0.3
        strategy._prune_parameters(
            graph, scope, [param], [0.3], place, lazy=True)
        self.assertTrue(len(strategy.backup.keys()) == 3)
        self.assertTrue('conv2d_3.w_0' in strategy.backup.keys())
        self.assertTrue('conv2d_3.b_0' in strategy.backup.keys())
        self.assertTrue('conv2d_4.w_0' in strategy.backup.keys())

        # pruning conv_4
        strategy.backup = {}
        param = graph.get_var('conv2d_4.w_0')
        ratio = 0.3
        strategy._prune_parameters(
            graph, scope, [param], [0.3], place, lazy=True)
        self.assertTrue(len(strategy.backup.keys()) == 5)
        self.assertTrue('conv2d_4.w_0' in strategy.backup.keys())
        self.assertTrue('conv2d_4.b_0' in strategy.backup.keys())
        self.assertTrue('conv2d_5.w_0' in strategy.backup.keys())
        self.assertTrue('conv2d_5.b_0' in strategy.backup.keys())
        self.assertTrue('conv2d_6.w_0' in strategy.backup.keys())

        # pruning conv_6
        strategy.backup = {}
        param = graph.get_var('conv2d_6.w_0')
        ratio = 0.3
        strategy._prune_parameters(
            graph, scope, [param], [0.3], place, lazy=True)
        self.assertTrue(len(strategy.backup.keys()) == 3)
        self.assertTrue('conv2d_6.w_0' in strategy.backup.keys())
        self.assertTrue('conv2d_6.b_0' in strategy.backup.keys())
        self.assertTrue('fc_0.w_0' in strategy.backup.keys())

    def test_prune_parameters(self):
        graph, scope, exe, place, loss = self.def_graph()
        pruning_axis = {'*': 0}
        criterions = {'*': 'l1_norm'}
        pruner = StructurePruner(None, pruning_axis, criterions)
        strategy = SensitivePruneStrategy(pruner=pruner)

        # pruning conv_1
        param = graph.get_var('conv2d_1.w_0')
        ratio = 0.3
        conv_1_w = scope.find_var('conv2d_1.w_0').get_tensor()
        conv_1_b = scope.find_var('conv2d_1.b_0').get_tensor()
        conv_2_w = scope.find_var('conv2d_2.w_0').get_tensor()
        conv_1_w_np = np.array(conv_1_w)
        mask = np.ones(conv_1_w_np.shape[0], dtype=bool)
        mask[np.sum(np.abs(conv_1_w_np), axis=(1, 2, 3)).argsort()[0]] = False
        pruned_conv_1_w = conv_1_w_np[mask]
        pruned_conv_1_b = np.array(conv_1_b)[mask]
        mask = 4 * [True] + list(mask)
        pruned_conv_2_w = np.array(conv_2_w)[:, mask, :, :]
        strategy._prune_parameters(graph, scope, [param], [0.3], place)
        self.assertTrue(np.allclose(pruned_conv_1_w, np.array(conv_1_w)))
        self.assertTrue(np.allclose(pruned_conv_1_b, np.array(conv_1_b)))
        self.assertTrue(np.allclose(pruned_conv_2_w, np.array(conv_2_w)))


if __name__ == '__main__':
    unittest.main()
