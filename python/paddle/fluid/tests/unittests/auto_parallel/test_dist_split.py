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
<<<<<<< HEAD

import paddle
from paddle.distributed.fleet import auto

=======
import paddle
from paddle.distributed.fleet import auto

from paddle.fluid import program_guard
from paddle.distributed.auto_parallel.utils import print_program_with_dist_attr

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
paddle.enable_static()


def make_program_dp2():
    main_program = paddle.fluid.Program()
    start_program = paddle.fluid.Program()
    with paddle.static.program_guard(main_program, start_program):
        x = paddle.static.data(name='x', shape=[4, 12, 16], dtype='float32')
        x.stop_gradient = False
<<<<<<< HEAD
        auto.shard_tensor(
            x, auto.ProcessMesh([0, 1], dim_names=["x"]), ["x", None, None]
        )
=======
        auto.shard_tensor(x, auto.ProcessMesh([0, 1], dim_names=["x"]),
                          ["x", None, None])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        out0, out1, out2 = paddle.split(x, num_or_sections=3, axis=1)
    return main_program, start_program


def parallelizer(program_func, rank):
    from paddle.distributed.auto_parallel.completion import Completer
<<<<<<< HEAD
    from paddle.distributed.auto_parallel.dist_context import DistributedContext
    from paddle.distributed.auto_parallel.partitioner import Partitioner
=======
    from paddle.distributed.auto_parallel.partitioner import Partitioner
    from paddle.distributed.auto_parallel.dist_context import DistributedContext
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    main_program, start_program = program_func()

    dist_context = DistributedContext()
    completer = Completer(dist_context)
    completer.complete_forward_annotation(main_program)
    dist_context.block_state.parse_forward_blocks(main_program)

    partitioner = Partitioner(dist_context, rank)
<<<<<<< HEAD
    dist_main_prog, _, _ = partitioner.partition(
        main_program, start_program, []
    )
=======
    dist_main_prog, _, _ = partitioner.partition(main_program, start_program,
                                                 [])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    return dist_main_prog, dist_context


class TestDistSplit(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_dist_split_dp2(self):

        for rank in range(2):
            dist_main_prog, dist_context = parallelizer(make_program_dp2, rank)
            ops = dist_main_prog.global_block().ops
            op_dist_attr = dist_context.get_op_dist_attr_for_program(ops[0])
            assert op_dist_attr.impl_type == "split"
            assert op_dist_attr.impl_idx == 0


if __name__ == "__main__":
    unittest.main()
