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
Single host:
python3.8 -m paddle.distributed.launch \
--devices=4 \
ipu \
--hosts=localhost \
--nproc_per_host=2 \
--ipus_per_replica=1 \
--ipu_partition=pod64 \
--vipu_server=10.137.96.62 \
test/ipu/distributed/test_dist_sample.py
Equal to:
poprun \
--host=localhost \
--num-instances=2 \
--num-replicas=4 \
--ipus-per-replica=1 \
--print-topology=yes \
python3.8 test/ipu/distributed/test_dist_sample.py
'''
'''
Multi hosts:
python3.8 -m paddle.distributed.launch \
--devices=4 \
ipu \
--hosts=host1,host2 \
--nproc_per_host=1 \
--ipus_per_replica=1 \
--ipu_partition=pod64 \
--vipu_server=10.137.96.62 \
test/ipu/distributed/test_dist_sample.py
Equal to:
poprun \
--host=host1,host2 \
--num-instances=2 \
--num-replicas=4 \
--ipus-per-replica=1 \
--print-topology=yes \
python3.8 test/ipu/distributed/test_dist_sample.py
'''

import os
import sys

import numpy as np

import paddle

mpi_comm = None


def Test(use_dist, file_name):
    paddle.enable_static()

    attrs = {"size": [128, 16], "padding_idx": -1, "dtype": 'float32'}

    scope = paddle.base.core.Scope()
    main_prog = paddle.static.Program()
    startup_prog = paddle.static.Program()
    paddle.seed(42)

    with paddle.base.scope_guard(scope):
        with paddle.static.program_guard(main_prog, startup_prog):
            x = paddle.static.data(name="x", shape=[3, 2, 1], dtype='int64')

            out = paddle.static.nn.embedding(x, **attrs)
            loss = paddle.mean(out)
            opt = paddle.optimizer.Adam(learning_rate=1e-1)
            opt.minimize(loss)

            feed_list = ["x"]
            fetch_list = [loss.name]

            place = paddle.IPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)

            ipu_strategy = paddle.static.IpuStrategy()
            if use_dist:
                ipu_strategy.set_graph_config(num_ipus=2, is_training=True)
                # Set distributed envs
                ipu_strategy.set_options(
                    {
                        "enable_distribution": True,
                        "enable_replicated_graphs": True,
                        "replicated_graph_count": 2,
                        "enable_distributed_replicated_graphs": True,
                        "global_replica_offset": int(
                            os.environ.get("PADDLE_TRAINER_ID")
                        )
                        * 2,
                        "global_replication_factor": 4,
                    }
                )
            else:
                ipu_strategy.set_graph_config(num_ipus=4, is_training=True)
                ipu_strategy.set_options(
                    {
                        "enable_replicated_graphs": True,
                        "replicated_graph_count": 4,
                    }
                )

            ipu_program = paddle.static.IpuCompiledProgram(
                main_prog, ipu_strategy=ipu_strategy
            )
            program = ipu_program.compile(feed_list, fetch_list)

            if use_dist:
                if os.environ.get("PADDLE_TRAINER_ID") == "0":
                    input_data = np.concatenate(
                        [
                            np.array(
                                [[[1], [3]], [[2], [4]], [[4], [127]]]
                            ).astype(np.int32),
                            np.array(
                                [[[1], [3]], [[2], [4]], [[4], [127]]]
                            ).astype(np.int32),
                        ]
                    )
                else:
                    input_data = np.concatenate(
                        [
                            np.array(
                                [[[8], [60]], [[50], [77]], [[90], [13]]]
                            ).astype(np.int32),
                            np.array(
                                [[[8], [60]], [[50], [77]], [[90], [13]]]
                            ).astype(np.int32),
                        ]
                    )
            else:
                input_data = np.concatenate(
                    [
                        np.array([[[1], [3]], [[2], [4]], [[4], [127]]]).astype(
                            np.int32
                        ),
                        np.array([[[1], [3]], [[2], [4]], [[4], [127]]]).astype(
                            np.int32
                        ),
                        np.array(
                            [[[8], [60]], [[50], [77]], [[90], [13]]]
                        ).astype(np.int32),
                        np.array(
                            [[[8], [60]], [[50], [77]], [[90], [13]]]
                        ).astype(np.int32),
                    ]
                )
            feed_data = {"x": input_data}

            for step in range(10):
                res = exe.run(program, feed=feed_data, fetch_list=fetch_list)

            if use_dist:
                res = mpi_comm.gather(res)
                if os.getenv("PADDLE_TRAINER_ID") == "0":
                    np.savetxt(file_name, np.array(res).flatten())
            else:
                np.savetxt(file_name, np.array(res).flatten())


if __name__ == "__main__":
    file_name = sys.argv[1]

    use_dist = False
    if 'PADDLE_TRAINER_ID' in os.environ:
        from mpi4py import MPI

        DISTRIBUTED_COMM = MPI.COMM_WORLD

        def _get_comm():
            global DISTRIBUTED_COMM
            if DISTRIBUTED_COMM is None:
                raise RuntimeError(
                    "Distributed Communication not setup. Please run setup_comm(MPI.COMM_WORLD) first."
                )
            return DISTRIBUTED_COMM

        mpi_comm = _get_comm()
        use_dist = True

    Test(use_dist, file_name)
