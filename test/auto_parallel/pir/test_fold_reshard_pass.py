# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import paddle
import paddle.distributed as dist
from paddle.distributed.auto_parallel.static.pir_pass import ReshardPasses

paddle.enable_static()

BATCH_SIZE = 2
SEQ_LEN = 4
HIDDEN_SIZE = 8
MP_SIZE = 2


class TestFoldReshardPass(unittest.TestCase):
    def test_base(self):
        main_program = paddle.base.Program()
        start_program = paddle.base.Program()
        with paddle.base.program_guard(main_program, start_program):
            mesh = dist.ProcessMesh([0, 1], dim_names=['x'])
            input = paddle.static.data(
                name='input', shape=[BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]
            )
            dist_input = dist.shard_tensor(input, mesh, [dist.Replicate()])
            x1 = dist.reshard(dist_input, mesh, [dist.Shard(0)])
            y1 = dist.reshard(dist_input, mesh, [dist.Shard(0)])
            z = x1 + y1
        reshard_op_num = 0

        for op in main_program.global_block().ops:
            if op.name() == "dist_op.reshard":
                reshard_op_num += 1
        self.assertEqual(reshard_op_num, 2)
        ReshardPasses.fold_reshard_pass(main_program)
        reshard_op_num = 0
        for op in main_program.global_block().ops:
            if op.name() == "dist_op.reshard":
                reshard_op_num += 1
        self.assertEqual(reshard_op_num, 1)
