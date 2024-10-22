# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import random
import sys
import unittest

import numpy as np

sys.path.append("../../auto_parallel")

from get_gpt_model import FakeDataset, generate_model

import paddle

paddle.enable_static()
from paddle import _legacy_C_ops
from paddle.distributed.fleet import auto


def dy_broadcast_helper(tensor):
    tensor = paddle._C_ops.broadcast(tensor, 0, 1)
    _legacy_C_ops.c_sync_calc_stream(tensor, tensor)
    return tensor


def apply_pass(use_recompute=False, no_recompute_segments=[]):
    strategy = auto.Strategy()
    strategy.auto_mode = "semi"
    strategy.reinit = True
    if use_recompute:
        recompute = strategy.recompute
        recompute.enable = True
        recompute.no_recompute_segments = no_recompute_segments
    return strategy


def reset_prog():
    paddle.base.framework.switch_main_program(paddle.static.Program())
    paddle.base.framework.switch_startup_program(paddle.static.Program())


class TestRandomControl(unittest.TestCase):
    def setUp(self):
        self.rtol = 1e-6
        self.atol = 1e-8
        self.batch_size = 1
        self.batch_num = 10
        self.clip_norm = 0.2
        self.dataset = FakeDataset(self.batch_size * self.batch_num)
        paddle.distributed.auto_parallel.parallel_manual_seed(100)

    def init(self, engine):
        paddle.seed(2022)
        np.random.seed(2022)
        random.seed(2022)
        place = paddle.base.CUDAPlace(paddle.distributed.ParallelEnv().dev_id)
        engine._executor = paddle.static.Executor(place)

    def get_engine(self, use_recompute=False, no_recompute_segments=[]):
        reset_prog()

        strategy = apply_pass(use_recompute, no_recompute_segments)
        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=0.00001, grad_clip=clip)
        model, loss = generate_model("mp", dropout_prob=0.1)

        engine = auto.Engine(model, loss, opt, strategy=strategy)
        self.init(engine)
        return engine

    def compare_mask_between_ranks(
        self, rank, mask_np_list, compare_idx, equal
    ):
        for np_mask in [mask_np_list[i] for i in compare_idx]:
            mask_tensor_local = paddle.to_tensor([np_mask.astype("float32")])
            if rank == 0:
                mask_tensor_remote = paddle.ones_like(mask_tensor_local)
                mask_tensor_remote = dy_broadcast_helper(mask_tensor_remote)
                if equal:
                    np.testing.assert_array_equal(
                        mask_tensor_remote.numpy(), mask_tensor_local.numpy()
                    )
                else:
                    assert not np.array_equal(
                        mask_tensor_remote.numpy(),
                        mask_tensor_local.numpy(),
                    )
            else:
                dy_broadcast_helper(mask_tensor_local)

    def test_random_ctrl_vanilla(self):
        # mp2 recompute training
        rc_engine = self.get_engine(False)
        train_dataloader = rc_engine.dataloader(
            self.dataset,
            batch_size=self.batch_size,
            mode="train",
            sample_split=3,
        )

        rc_engine.prepare(mode="train")
        mask_name_list = [f'dropout_{i}.tmp_1' for i in range(7)]
        mask_var_list = [
            rc_engine.main_program.global_block().var(varname)
            for varname in mask_name_list
        ]

        for data in train_dataloader:
            outs = rc_engine.run(data, fetch_list=mask_var_list, mode="train")
        mask_np_list = [outs['fetches'][varname] for varname in mask_name_list]

        paddle.disable_static()
        rank = paddle.distributed.get_rank()
        # check global mask consistent across ranks
        global_index = [0, 2, 3, 5, 6]
        self.compare_mask_between_ranks(
            rank, mask_np_list, global_index, equal=True
        )
        local_index = [1, 4]
        # check local mask different across ranks
        self.compare_mask_between_ranks(
            rank, mask_np_list, local_index, equal=False
        )
        paddle.enable_static()

        # check program
        ops = rc_engine.main_program.global_block().ops
        rng_names = []
        seed_var_names = []
        for op in ops:
            if op.type == "seed":
                rng_names.append(op.attr('rng_name'))
            if op.type == "dropout":
                seed_var_names.append(op.input("Seed")[0])
        rank = paddle.distributed.get_rank()

        self.assertEqual(
            rng_names,
            [
                'mesh:1_dim0:-1',
                f'mesh:1_dim0:{rank}',
                'mesh:1_dim0:-1',
                'mesh:1_dim0:-1',
                f'mesh:1_dim0:{rank}',
                'mesh:1_dim0:-1',
                'mesh:1_dim0:-1',
            ],
        )
        self.assertEqual(
            seed_var_names,
            [
                'tensor_parallel_seed.tmp_0',
                'tensor_parallel_seed.tmp_1',
                'tensor_parallel_seed.tmp_2',
                'tensor_parallel_seed.tmp_3',
                'tensor_parallel_seed.tmp_4',
                'tensor_parallel_seed.tmp_5',
                'tensor_parallel_seed.tmp_6',
            ],
        )

    def test_random_ctrl_with_recompute(self):
        # mp2 recompute training
        rc_engine = self.get_engine(True)
        train_dataloader = rc_engine.dataloader(
            self.dataset,
            batch_size=self.batch_size,
            mode="train",
            sample_split=3,
        )

        rc_engine.prepare(mode="train")
        mask_name_list = [f'dropout_{i}.tmp_1' for i in range(7)]
        recompute_mask_name_list = [
            'dropout_0.tmp_1.subprog_1',
            'dropout_1.tmp_1.subprog_1',
            'dropout_2.tmp_1.subprog_1',
            'dropout_3.tmp_1.subprog_1',
            'dropout_4.tmp_1.subprog_0',
            'dropout_5.tmp_1.subprog_0',
            'dropout_6.tmp_1.subprog_0',
        ]
        mask_var_list = [
            rc_engine.main_program.global_block().var(varname)
            for varname in mask_name_list + recompute_mask_name_list
        ]

        for data in train_dataloader:
            outs = rc_engine.run(data, fetch_list=mask_var_list, mode="train")
        mask_np_list = [
            outs['fetches'][varname]
            for varname in mask_name_list + recompute_mask_name_list
        ]

        # check recompute is mask the same within local device
        for i in range(7):
            mask_fw = mask_np_list[i].astype("float32")
            mask_rc = mask_np_list[i + 7].astype("float32")
            np.testing.assert_array_equal(
                mask_fw,
                mask_rc,
            )

        paddle.disable_static()
        # check global mask consistent across ranks
        rank = paddle.distributed.get_rank()
        global_index = [0, 2, 3, 5, 6]
        self.compare_mask_between_ranks(
            rank, mask_np_list, global_index, equal=True
        )
        local_index = [1, 4]
        # check local mask different across ranks
        self.compare_mask_between_ranks(
            rank, mask_np_list, local_index, equal=False
        )
        paddle.enable_static()

        # check program
        rank = paddle.distributed.get_rank()
        ops = rc_engine.main_program.global_block().ops
        rng_names = []
        seed_var_names = []
        for op in ops:
            if op.type == "seed":
                rng_names.append(op.attr('rng_name'))
            if op.type == "dropout":
                seed_var_names.append(op.input("Seed")[0])

        self.assertEqual(
            rng_names,
            [
                'mesh:1_dim0:-1',
                f'mesh:1_dim0:{rank}',
                'mesh:1_dim0:-1',
                'mesh:1_dim0:-1',
                f'mesh:1_dim0:{rank}',
                'mesh:1_dim0:-1',
                'mesh:1_dim0:-1',
            ],
        )
        self.assertEqual(
            seed_var_names,
            [
                'rc_seed_0.tmp_0',
                'rc_seed_1.tmp_0',
                'rc_seed_2.tmp_0',
                'rc_seed_3.tmp_0',
                'rc_seed_4.tmp_0',
                'rc_seed_5.tmp_0',
                'rc_seed_6.tmp_0',
                'rc_seed_4.tmp_0',
                'rc_seed_5.tmp_0',
                'rc_seed_6.tmp_0',
                'rc_seed_0.tmp_0',
                'rc_seed_1.tmp_0',
                'rc_seed_2.tmp_0',
                'rc_seed_3.tmp_0',
            ],
        )


if __name__ == "__main__":
    unittest.main()
