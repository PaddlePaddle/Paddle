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

import unittest
import paddle
from paddle.distributed.fleet import auto

from paddle.fluid import program_guard
from paddle.fluid.backward import append_backward
from paddle.distributed.auto_parallel.utils import print_program_with_dist_attr
from test_dist_pnorm import parallelizer

paddle.enable_static()


def make_program_lookup_table_v1_mp_dp():
    main_program = paddle.fluid.Program()
    start_program = paddle.fluid.Program()
    block = main_program.global_block()
    with paddle.static.program_guard(main_program, start_program):

        src_ids = paddle.static.data(name='src_ids',
                                     shape=[12, 512, 1],
                                     dtype='int64')
        src_ids.stop_gradient = True
        emb_out = paddle.fluid.layers.embedding(
            input=src_ids,
            size=[64, 128],
            param_attr=paddle.fluid.ParamAttr(name="emb_weight"),
            dtype="float32",
            is_sparse=False)
        loss = paddle.fluid.layers.reduce_mean(emb_out)

        auto.shard_tensor(
            src_ids, auto.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"]),
            ["x", None, None])
        emb_weight = block.vars["emb_weight"]
        auto.shard_tensor(
            emb_weight, auto.ProcessMesh([[0, 1], [2, 3]],
                                         dim_names=["x", "y"]), ["y", None])

    return main_program, start_program, loss


class TestDistPNorm(unittest.TestCase):

    def test_lookup_table_v1_mp_dp(self):

        for rank in range(4):
            dist_main_prog, dist_context = parallelizer(
                make_program_lookup_table_v1_mp_dp, rank)
            ops = dist_main_prog.global_block().ops

            op_types = []
            for op in ops:
                op_types.append(op.type)

            assert op_types == [
                'reshape2', 'c_embedding', 'c_allreduce_sum', 'reduce_mean',
                'fill_constant', 'reduce_mean_grad', 'c_identity',
                'c_embedding_grad', 'c_allreduce_sum', 'scale'
            ]


if __name__ == "__main__":
    unittest.main()
