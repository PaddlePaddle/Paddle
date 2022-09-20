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

paddle.enable_static()


def make_program_dp2():
    main_program = paddle.fluid.Program()
    start_program = paddle.fluid.Program()
    with paddle.static.program_guard(main_program, start_program):
        x = paddle.static.data(name='x', shape=[4, 4, 8], dtype='float32')
        x.stop_gradient = False
        auto.shard_tensor(x, auto.ProcessMesh([0, 1], dim_names=["x"]),
                          ["x", None, None])

        tmp_0 = paddle.reshape(x, shape=[0, 0, 4, 2])
        tmp_1 = paddle.reshape(tmp_0, shape=[0, 0, 8])
        tmp_2 = tmp_1.reshape((tmp_1.shape[0], tmp_1.shape[1], -1))
    return main_program, start_program


def parallelizer(program_func, rank):
    from paddle.distributed.auto_parallel.completion import Completer
    from paddle.distributed.auto_parallel.partitioner import Partitioner
    from paddle.distributed.auto_parallel.dist_context import DistributedContext

    main_program, start_program = program_func()

    dist_context = DistributedContext()
    completer = Completer(dist_context)
    completer.complete_forward_annotation(main_program)
    dist_context.block_state.parse_forward_blocks(main_program)

    partitioner = Partitioner(dist_context, rank)
    dist_main_prog, _, _ = partitioner.partition(main_program, start_program,
                                                 [])

    return dist_main_prog, dist_context


class TestDistReshape(unittest.TestCase):

    def test_dist_reshape_mp2(self):

        for rank in range(2):
            dist_main_prog, dist_context = parallelizer(make_program_dp2, rank)
            ops = dist_main_prog.global_block().ops
            for idx, op in enumerate(ops):
                op_dist_attr = dist_context.get_op_dist_attr_for_program(op)
                assert op_dist_attr.impl_type == "reshape2"
                assert op_dist_attr.impl_idx == idx

                if op_dist_attr.impl_idx == 2:
                    assert op.desc.attr('shape')[0] == 2


if __name__ == "__main__":
    unittest.main()
