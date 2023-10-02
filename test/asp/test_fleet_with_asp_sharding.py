# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2021 NVIDIA Corporation.  All rights reserved.
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

import os
import unittest

import numpy as np

import paddle
from paddle import base
from paddle.distributed import fleet
from paddle.incubate import asp as sparsity
from paddle.incubate.asp import ASPHelper

cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
if cuda_visible_devices is None or cuda_visible_devices == "":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices.split(',')[0]

paddle.enable_static()


class TestFleetWithASPSharding(unittest.TestCase):
    def setUp(self):
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36213"
        os.environ["PADDLE_CURRENT_ENDPOINTS"] = "127.0.0.1:36213"
        os.environ["PADDLE_TRAINERS_NUM"] = "1"
        os.environ["PADDLE_TRAINER_ID"] = "0"

        os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = "0.1"
        os.environ['FLAGS_sync_nccl_allreduce'] = "1"
        os.environ['FLAGS_eager_delete_tensor_gb'] = "0"
        os.environ['FLAGS_fuse_parameter_memory_size'] = "32"
        os.environ['FLAGS_fuse_parameter_groups_size'] = "50"
        os.environ['FLAGS_check_nan_inf'] = "0"

    def net(self, main_prog, startup_prog):
        with base.program_guard(main_prog, startup_prog):
            input_x = paddle.static.data(
                name="x", shape=[-1, 32], dtype='float32'
            )
            input_y = paddle.static.data(name="y", shape=[-1, 1], dtype='int64')

            fc_1 = paddle.static.nn.fc(x=input_x, size=64, activation='tanh')
            fc_2 = paddle.static.nn.fc(x=fc_1, size=64, activation='tanh')
            fc_3 = paddle.static.nn.fc(x=fc_2, size=64, activation='tanh')
            fc_4 = paddle.static.nn.fc(x=fc_3, size=64, activation='tanh')
            prediction = paddle.static.nn.fc(
                x=fc_4, size=2, activation='softmax'
            )
            cost = paddle.nn.functional.cross_entropy(
                input=prediction,
                label=input_y,
                reduction='none',
                use_softmax=False,
            )
            avg_cost = paddle.mean(x=cost)

            dist_strategy = paddle.distributed.fleet.DistributedStrategy()
            dist_strategy.sharding = True
            dist_strategy.sharding_configs = {
                "sharding_segment_strategy": "segment_broadcast_MB",
                "segment_broadcast_MB": 32,
                "segment_anchors": None,
                "sharding_degree": 8,
                "mp_degree": 1,
                "hybrid_dp": False,
                "gradient_merge_acc_step": 1,
            }
            dist_strategy.nccl_comm_num = 1
            dist_strategy.asp = True
        return avg_cost, dist_strategy, input_x, input_y

    def test_with_asp_sharding(self):
        fleet.init(is_collective=True)
        train_prog, startup_prog = base.Program(), base.Program()
        avg_cost, strategy, input_x, input_y = self.net(
            train_prog, startup_prog
        )

        with base.program_guard(train_prog, startup_prog):
            optimizer = paddle.optimizer.SGD(learning_rate=0.01)
            optimizer = fleet.distributed_optimizer(
                optimizer, strategy=strategy
            )
            optimizer.minimize(avg_cost)

        if paddle.base.is_compiled_with_cuda():
            place = base.CUDAPlace(
                int(os.environ.get('FLAGS_selected_gpus', 0))
            )
        else:
            place = base.CPUPlace()

        exe = base.Executor(place)
        feeder = base.DataFeeder(feed_list=[input_x, input_y], place=place)
        exe.run(startup_prog)

        sparsity.prune_model(train_prog)

        data = (np.random.randn(64, 32), np.random.randint(2, size=(64, 1)))
        exe.run(train_prog, feed=feeder.feed([data]))

        for param in train_prog.global_block().all_parameters():
            if ASPHelper._is_supported_layer(train_prog, param.name):
                mat = np.array(
                    base.global_scope().find_var(param.name).get_tensor()
                )
                if (len(param.shape) == 4 and param.shape[1] < 4) or (
                    len(param.shape) == 2 and param.shape[0] < 4
                ):
                    self.assertFalse(
                        paddle.incubate.asp.check_sparsity(mat.T, n=2, m=4)
                    )
                else:
                    self.assertTrue(
                        paddle.incubate.asp.check_sparsity(mat.T, n=2, m=4)
                    )


if __name__ == "__main__":
    unittest.main()
