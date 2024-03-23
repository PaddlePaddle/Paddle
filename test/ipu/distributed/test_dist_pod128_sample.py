#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''
python3.8 -m paddle.distributed.launch \
--devices=128 \
ipu \
--hosts=host1,host2 \
--ipus_per_host=2 \
--nproc_per_host=1 \
--ipu_partition=pod128 \
--vipu_server=lr17-1-ctrl \
test/ipu/disabled/test_dist_pod128_ipu.py
Equal to:
poprun \
--host=localhost,host2 \
--num-instances=2 \
--num-replicas=64 \
--ipus-per-replica=2 \
--print-topology=yes \
--vipu-partition=pod128_bert \
--vipu-server-host=lr17-1-ctrl \
--update-partition=yes \
python3.8 test/ipu/disabled/test_dist_pod128_ipu.py
'''

import os

import numpy as np

import paddle


def TestDistTraining():
    paddle.enable_static()

    attrs = {"size": [128, 16], "padding_idx": -1, "dtype": 'float32'}

    scope = paddle.base.core.Scope()
    main_prog = paddle.static.Program()
    startup_prog = paddle.static.Program()
    paddle.seed(42)

    np.random.seed(42)
    input_data = np.random.uniform(0, 127, size=[128, 3, 2, 1]).astype(np.int32)

    with paddle.base.scope_guard(scope):
        with paddle.static.program_guard(main_prog, startup_prog):
            x = paddle.static.data(name="x", shape=[3, 2, 1], dtype='int64')
            with paddle.static.ipu_shard_guard(index=0, stage=0):
                out = paddle.static.nn.embedding(x, **attrs)
            with paddle.static.ipu_shard_guard(index=1, stage=1):
                loss = paddle.mean(out)
            opt = paddle.optimizer.Adam(learning_rate=1e-1)
            opt.minimize(loss)

            feed_list = ["x"]
            fetch_list = [loss.name]

            place = paddle.IPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)

            ipu_strategy = paddle.static.IpuStrategy()
            ipu_strategy.set_graph_config(
                num_ipus=64, is_training=True, enable_manual_shard=True
            )
            ipu_strategy.set_pipelining_config(
                enable_pipelining=True,
                batches_per_step=1,
                enable_gradient_accumulation=True,
                accumulation_factor=4,
            )
            ipu_strategy.set_options(
                {
                    "enable_distribution": True,
                    "enable_replicated_graphs": True,
                    "replicated_graph_count": 32,
                    "enable_distributed_replicated_graphs": True,
                    "global_replica_offset":
                    # Paddle : int(os.environ.get("PADDLE_TRAINER_ID")) * 32
                    # PopRun : int(os.environ.get("POPDIST_REPLICA_INDEX_OFFSET"))
                    int(os.environ.get("PADDLE_TRAINER_ID")) * 32,
                    "global_replication_factor": 64,
                    "location_optimizer": {
                        "on_chip": False,
                        "use_replicated_tensor_sharding": True,
                    },
                }
            )

            ipu_program = paddle.static.IpuCompiledProgram(
                main_prog, ipu_strategy=ipu_strategy
            )
            program = ipu_program.compile(feed_list, fetch_list)

            for i in range(10):
                res = exe.run(
                    program, feed={"x": input_data}, fetch_list=fetch_list
                )
                print(f"index: {i}, result: {res}")


if __name__ == "__main__":
    TestDistTraining()
