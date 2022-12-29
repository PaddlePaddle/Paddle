# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import random
import sys
import unittest

import numpy as np
from auto_parallel_pass_test_base import AutoPallelPassTestBase

import paddle
import paddle.distributed.fleet as fleet
from paddle.distributed.auto_parallel.dist_context import (
    get_default_distributed_context,
)
from paddle.distributed.auto_parallel.process_group import (
    get_all_process_groups,
)
from paddle.distributed.passes import PassContext, new_pass

sys.path.append("..")


class TestShardingPass(AutoPallelPassTestBase):
    def init(self):
        if paddle.is_compiled_with_cuda():
            paddle.set_flags({'FLAGS_cudnn_deterministic': 1})
        self.rtol = 1e-5
        self.atol = 1e-8

        rank = paddle.distributed.get_rank()
        paddle.seed(rank + 2021)
        random.seed(rank + 2021)
        np.random.seed(rank + 2021)

    def apply_passes(self):
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.semi_auto = True
        dist_strategy.sharding = True
        dist_strategy.sharding_configs = {
            "sharding_degree": 2,
            "stage": 2,
        }
        fleet.init(is_collective=True, strategy=dist_strategy)

    def apply_no_passes(self):
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.pipeline = False
        dist_strategy.recompute = False
        dist_strategy.semi_auto = True
        fleet.init(is_collective=True, strategy=dist_strategy)

    def test_bs_8(self):
        self.check_main(
            gpus=[0, 1], batch_size=8, sequence_len=512, vocab_size=1000
        )

    def get_model(self, place, batch_size, sequence_len, vocab_size):
        return self.get_gpt_model(
            'dp', place, batch_size, sequence_len, vocab_size
        )


class TestShardingStage2WithNewEXE(AutoPallelPassTestBase):
    def init(self):
        if paddle.is_compiled_with_cuda():
            paddle.set_flags({'FLAGS_cudnn_deterministic': 1})
        self.rtol = 1e-5
        self.atol = 1e-8

        rank = paddle.distributed.get_rank()
        paddle.seed(rank + 2021)
        random.seed(rank + 2021)
        np.random.seed(rank + 2021)

    def apply_passes(self):
        os.environ["FLAGS_CONVERT_GRAPH_TO_PROGRAM"] = str(1)
        os.environ["FLAGS_add_dependency_for_communication_op"] = 'false'
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.semi_auto = True
        dist_strategy.recompute = True
        dist_strategy.recompute_configs = {"checkpoints": ["tmp_3", "tmp_6"]}
        fleet.init(is_collective=True, strategy=dist_strategy)
        self._apply_pass = True

    def apply_no_passes(self):
        os.environ["FLAGS_CONVERT_GRAPH_TO_PROGRAM"] = str(0)
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.semi_auto = True
        dist_strategy.recompute = True
        dist_strategy.recompute_configs = {"checkpoints": ["tmp_3", "tmp_6"]}
        fleet.init(is_collective=True, strategy=dist_strategy)
        self._apply_pass = False

    def test_bs_8(self):
        self.check_main(
            gpus=[0, 1], batch_size=8, sequence_len=512, vocab_size=1000
        )

    # Hack to apply pass
    def get_model(
        self,
        place,
        batch_size,
        sequence_len,
        vocab_size,
    ):

        (
            dist_main_prog,
            dist_startup_prog,
            data_holder,
            [loss],
            gen_data,
            params_grads,
        ) = self.get_gpt_model(
            'dp',
            place,
            batch_size,
            sequence_len,
            vocab_size,
            need_params_grads=True,
        )

        if self._apply_pass:
            cur_rank = paddle.distributed.get_rank()
            config = {}
            config["dist_context"] = get_default_distributed_context()
            config["global_rank"] = cur_rank
            config["stage"] = 2
            config["degree"] = 2
            config["sharding_degree"] = 2
            config["enable_overlap"] = True
            config["param_comm_stream_num"] = 2
            config["grad_comm_stream_num"] = 2
            config["param_bucket_size_numel"] = 1024 * 1024
            config["grad_bucket_size_numel"] = 1024 * 1024
            config["partition_algor"] = 'use_order'
            config["enable_hierarchical_comm"] = False
            config["params_grads"] = params_grads

            pass1 = new_pass("auto_parallel_sharding", config)
            pass1.apply([dist_main_prog], [dist_startup_prog], PassContext())

            pass2 = new_pass(
                "auto_parallel_supplement_explicit_dependencies", config
            )
            pass2.apply([dist_main_prog], [dist_startup_prog], PassContext())

            for process_group in get_all_process_groups():
                if cur_rank not in process_group.ranks:
                    continue
                process_group.instantiate()

            with open(
                "./appled_program.txt.{}".format(paddle.distributed.get_rank()),
                "w+",
            ) as f:
                f.write(str(dist_main_prog))
        else:
            with open(
                "./vanilla_program.txt.{}".format(
                    paddle.distributed.get_rank()
                ),
                "w+",
            ) as f:
                f.write(str(dist_main_prog))

        return dist_main_prog, dist_startup_prog, data_holder, [loss], gen_data


if __name__ == "__main__":
    unittest.main()
