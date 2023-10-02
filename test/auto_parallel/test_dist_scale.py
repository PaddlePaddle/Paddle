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

paddle.enable_static()


def make_program():
    main_program = paddle.base.Program()
    start_program = paddle.base.Program()
    with paddle.static.program_guard(main_program, start_program):
        x = paddle.static.data(name='x', shape=[4, 4, 8], dtype='float32')
        x.stop_gradient = False
        auto.shard_tensor(
            x, auto.ProcessMesh([0, 1], dim_names=["x"]), [None, "x", None]
        )
        res = paddle.scale(x, scale=2.0, bias=1.0)
    return main_program, start_program


def parallelizer(program_func, rank):
    from paddle.distributed.auto_parallel.static.completion import Completer
    from paddle.distributed.auto_parallel.static.dist_context import (
        DistributedContext,
    )
    from paddle.distributed.auto_parallel.static.partitioner import Partitioner

    main_program, start_program = program_func()

    dist_context = DistributedContext()
    completer = Completer(dist_context)
    completer.complete_forward_annotation(main_program)
    dist_context.block_state.parse_forward_blocks(main_program)

    partitioner = Partitioner(dist_context, rank)
    dist_main_prog, _, _ = partitioner.partition(
        main_program, start_program, []
    )

    return dist_main_prog, dist_context


class TestDistScale(unittest.TestCase):
    def test_dist_scale(self):
        dist_main_prog, dist_context = parallelizer(make_program, 0)
        ops = dist_main_prog.global_block().ops
        scale_op = ops[0]
        dist_op = dist_context.get_dist_op_for_program(scale_op)
        assert dist_op.dist_attr.impl_type == "scale"
        assert dist_op.dist_attr.impl_idx == 0

        in_name = scale_op.input_arg_names[0]
        out_name = scale_op.output_arg_names[0]
        in_dims_mapping = dist_op.dist_attr.get_input_dims_mapping(in_name)
        out_dims_mapping = dist_op.dist_attr.get_output_dims_mapping(out_name)

        assert in_dims_mapping == out_dims_mapping


if __name__ == "__main__":
    unittest.main()
