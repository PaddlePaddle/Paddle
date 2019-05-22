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

import paddle
import paddle.fluid as fluid
import os

import dist_ctr_reader
from test_dist_base import TestDistRunnerBase, runtime_main, RUN_STEP

IS_SPARSE = True
os.environ['PADDLE_ENABLE_REMOTE_PREFETCH'] = "1"

# Fix seed for test
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1


class TestDistCTR2x2(TestDistRunnerBase):
    def run_pserver(self, args):
        self.get_model(batch_size=2)
        # NOTE: pserver should not call memory optimize
        t = self.get_transpiler(args.trainer_id,
                                fluid.default_main_program(), args.endpoints,
                                args.trainers, args.sync_mode, False,
                                args.current_endpoint)
        pserver_prog = t.get_pserver_program(args.current_endpoint)
        startup_prog = t.get_startup_program(args.current_endpoint,
                                             pserver_prog)

        model_dir = os.getenv("MODEL_DIR", "")

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_prog)
        exe.run(pserver_prog)

    def run_trainer(self, args):
        test_program, avg_cost, train_reader = self.get_model(batch_size=2)

        if args.mem_opt:
            fluid.memory_optimize(fluid.default_main_program(), skip_grads=True)
        if args.update_method == "pserver":
            t = self.get_transpiler(args.trainer_id,
                                    fluid.default_main_program(),
                                    args.endpoints, args.trainers,
                                    args.sync_mode)

            trainer_prog = t.get_trainer_program()
        else:
            trainer_prog = fluid.default_main_program()

        place = fluid.CPUPlace()
        startup_exe = fluid.Executor(place)
        startup_exe.run(fluid.default_startup_program())

        strategy = fluid.ExecutionStrategy()
        strategy.num_threads = 1
        strategy.allow_op_delay = False

        build_stra = fluid.BuildStrategy()

        if args.use_reduce:
            build_stra.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
        else:
            build_stra.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.AllReduce

        exe = fluid.ParallelExecutor(
            use_cuda=False,
            loss_name=avg_cost.name,
            exec_strategy=strategy,
            build_strategy=build_stra)

        feed_var_list = [
            var for var in trainer_prog.global_block().vars.values()
            if var.is_data
        ]

        feeder = fluid.DataFeeder(feed_var_list, place)
        reader_generator = train_reader()

        def get_data():
            origin_batch = next(reader_generator)
            if args.update_method == "pserver" and args.use_reader_alloc:
                new_batch = []
                for offset, item in enumerate(origin_batch):
                    if offset % 2 == args.trainer_id:
                        new_batch.append(item)
                return new_batch
            else:
                return origin_batch

        model_dir = os.getenv("MODEL_DIR", "")

        for _ in six.moves.xrange(RUN_STEP):
            exe.run(fetch_list=[avg_cost.name],feed=feeder.feed(get_data()))
        io.save_persistables(startup_exe, model_dir, trainer_prog)


    def get_model(self, batch_size=2):
        dnn_input_dim, lr_input_dim = dist_ctr_reader.load_data_meta()
        """ network definition """
        dnn_data = fluid.layers.data(
            name="dnn_data",
            shape=[-1, 1],
            dtype="int64",
            lod_level=1,
            append_batch_size=False)
        lr_data = fluid.layers.data(
            name="lr_data",
            shape=[-1, 1],
            dtype="int64",
            lod_level=1,
            append_batch_size=False)
        label = fluid.layers.data(
            name="click",
            shape=[-1, 1],
            dtype="int64",
            lod_level=0,
            append_batch_size=False)

        # build dnn model
        dnn_layer_dims = [128, 64, 32, 1]
        dnn_embedding = fluid.layers.embedding(
            is_distributed=False,
            input=dnn_data,
            size=[dnn_input_dim, dnn_layer_dims[0]],
            param_attr=fluid.ParamAttr(
                name="deep_embedding",
                initializer=fluid.initializer.Constant(value=0.01)),
            is_sparse=IS_SPARSE)
        dnn_pool = fluid.layers.sequence_pool(
            input=dnn_embedding, pool_type="sum")
        dnn_out = dnn_pool
        for i, dim in enumerate(dnn_layer_dims[1:]):
            fc = fluid.layers.fc(
                input=dnn_out,
                size=dim,
                act="relu",
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(value=0.01)),
                name='dnn-fc-%d' % i)
            dnn_out = fc

        # build lr model
        lr_embbding = fluid.layers.embedding(
            is_distributed=False,
            input=lr_data,
            size=[lr_input_dim, 1],
            param_attr=fluid.ParamAttr(
                name="wide_embedding",
                initializer=fluid.initializer.Constant(value=0.01)),
            is_sparse=IS_SPARSE)
        lr_pool = fluid.layers.sequence_pool(input=lr_embbding, pool_type="sum")

        merge_layer = fluid.layers.concat(input=[dnn_out, lr_pool], axis=1)

        predict = fluid.layers.fc(input=merge_layer, size=2, act='softmax')
        cost = fluid.layers.cross_entropy(input=predict, label=label)
        avg_cost = fluid.layers.mean(x=cost)

        sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.0001,
                                            regularization=regularization)
        sgd_optimizer.minimize(avg_cost)
        return fluid.default_main_program(), avg_cost, train_reader, predict


if __name__ == "__main__":
    runtime_main(TestDistCTR2x2)
