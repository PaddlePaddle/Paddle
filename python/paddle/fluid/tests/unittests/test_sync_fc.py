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
import numpy as np
import unittest
import paddle
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper, weight_vars_guard
import six
import paddle.fluid.core as core

GPUs = 1
if core.is_compiled_with_cuda():
    GPUs = core.get_cuda_device_count()
else:
    exit(0)
N = GPUs
F = 4
# size of FC should be times of GPUs
C = GPUs * 4


class TestSyncFC(unittest.TestCase):
    def setUp(self):
        self.fc_w0 = np.random.rand(F, C / GPUs).astype("float32")
        self.fc_w1 = self.fc_w0
        self.fc_w = np.concatenate([self.fc_w0 for _ in xrange(GPUs)], axis=1)
        print(self.fc_w.shape)

    def data_reader(self):
        np.random.seed(1)

        def reader():
            x_sequence = [
                np.random.rand(F).astype("float32")
                for _ in six.moves.xrange(100)
            ]
            label_sequence = [
                np.random.rand(1).astype("float32")
                for _ in six.moves.xrange(100)
            ]
            for x, label in zip(x_sequence, label_sequence):
                yield x, label

        return reader

    def fc_net(self, is_sync_fc=False, model_parallelism_weights=[]):
        train_prog = fluid.Program()
        startup_prog = fluid.Program()
        train_prog.random_seed = 1
        startup_prog.random_seed = 1
        model_parallel_weight = []

        with fluid.program_guard(train_prog, startup_prog):
            with fluid.unique_name.guard():
                x = fluid.layers.data("x", shape=[F], dtype="float32")
                y = fluid.layers.data("y", shape=[1], dtype="float32")

                if is_sync_fc:
                    with weight_vars_guard(model_parallel_weight):
                        y_predict = fluid.layers.fc(
                            x,
                            size=C / GPUs,
                            distributed=True,
                            trainers=1,
                            param_attr=fluid.ParamAttr(name="fc_w"),
                            bias_attr=fluid.ParamAttr(
                                name="fc_b",
                                initializer=fluid.initializer.Constant(0.1)))
                else:
                    y_predict = fluid.layers.fc(
                        x,
                        size=C,
                        param_attr=fluid.ParamAttr(name="fc_w"),
                        bias_attr=fluid.ParamAttr(
                            name="fc_b",
                            initializer=fluid.initializer.Constant(0.1)))
                cost = fluid.layers.square_error_cost(input=y_predict, label=y)
                avg_cost = fluid.layers.mean(x=cost)

                sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
                sgd_optimizer.minimize(avg_cost)

        return train_prog, startup_prog, avg_cost, model_parallel_weight

    def get_feeder(self, train_prog):
        feed_var_list = [
            var for var in train_prog.global_block().vars.values()
            if var.is_data
        ]
        return fluid.DataFeeder(feed_var_list, fluid.CUDAPlace(0))

    def train_sync_fc(self):
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            no_collect_weight = []
            train_prog, startup_prog, loss, weight_vars = self.fc_net(
                is_sync_fc=True, model_parallelism_weights=no_collect_weight)
            with open("train_prog", "w") as f:
                f.write(train_prog.to_string(True))
            place = fluid.CUDAPlace(0)
            startup_exe = fluid.Executor(place)
            startup_exe.run(startup_prog)
            feeder = self.get_feeder(train_prog)

            scope.var("fc_w").get_tensor().set(self.fc_w0, place)
            strategy = fluid.ExecutionStrategy()
            build_strategy = fluid.BuildStrategy()
            build_strategy.model_parallelism_weights = [
                w.name for w in weight_vars
            ]
            build_strategy.reduce_strategy = fluid.BuildStrategy(
            ).ReduceStrategy.AllReduce
            build_strategy.debug_graphviz_path = "./graph.dot"
            parallel_exe = fluid.ParallelExecutor(
                loss_name=loss.name,
                use_cuda=True,
                main_program=train_prog,
                exec_strategy=strategy,
                build_strategy=build_strategy)
            reader = paddle.batch(self.data_reader(), batch_size=2 * GPUs)
            data = next(reader())
            fetch_vars = [loss.name, "fc_w"]
            ret = parallel_exe.run(fetch_list=fetch_vars,
                                   feed=feeder.feed(data))
        return ret

    def train_norm_fc(self):
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            train_prog, startup_prog, loss, _ = self.fc_net(is_sync_fc=False)
            with open("train_prog_norm", "w") as f:
                f.write(train_prog.to_string(True))
            place = fluid.CUDAPlace(0)
            startup_exe = fluid.Executor(place)
            startup_exe.run(startup_prog)
            feeder = self.get_feeder(train_prog)
            scope.var("fc_w").get_tensor().set(self.fc_w, place)
            reader = paddle.batch(self.data_reader(), batch_size=2 * GPUs)
            data = next(reader())
            fetch_vars = [loss.name, "fc_w"]
            ret = startup_exe.run(train_prog,
                                  fetch_list=fetch_vars,
                                  feed=feeder.feed(data))
        return ret

    def test_sync_fc(self):
        if not core.is_compiled_with_cuda() and core.get_cuda_device_count(
        ) > 2:
            return

        loss, fc_w = self.train_norm_fc()
        loss_dist, fc_w_dist = self.train_sync_fc()
        self.assertAlmostEqual(loss, np.mean(loss_dist), delta=1e-5)
        merge_fc_w = np.concatenate(np.split(fc_w_dist, GPUs, axis=0), axis=1)
        self.assertTrue(
            np.allclose(
                merge_fc_w, fc_w, atol=1e-5, equal_nan=False))


if __name__ == "__main__":
    unittest.main()
