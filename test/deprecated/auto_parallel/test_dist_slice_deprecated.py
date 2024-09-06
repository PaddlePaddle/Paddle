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


def make_program_dp2():
    main_program = paddle.base.Program()
    start_program = paddle.base.Program()
    with paddle.static.program_guard(main_program, start_program):
        x = paddle.static.data(name='x', shape=[4, 5, 6], dtype='float32')
        auto.shard_tensor(
            x, auto.ProcessMesh([0, 1], dim_names=["x"]), ["x", None, None]
        )

        tmp_0 = x[0]
        tmp_1 = x[:, 0, :]
        tmp_2 = x[:, :, 1]
        tmp_3 = x[:2, :2, :2]
        tmp_3 = x[:4, :2, :2]
    return main_program, start_program


def make_program_serial():
    main_program = paddle.base.Program()
    start_program = paddle.base.Program()
    with paddle.static.program_guard(main_program, start_program):
        x = paddle.static.data(name='x', shape=[4, 5, 6], dtype='float32')
        auto.shard_tensor(
            x, auto.ProcessMesh([0], dim_names=["x"]), [None, None, None]
        )

        tmp_0 = x[0]
        tmp_1 = x[:, 0, :]
        tmp_2 = x[:, :, 1]
        tmp_3 = x[2, 2, :]
        tmp_4 = x[:2, :2, :2]
        tmp_5 = x[0, 0, 0]
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


class TestDistSlice(unittest.TestCase):
    def test_dist_slice_dp2(self):
        for rank in range(2):
            dist_main_prog, dist_context = parallelizer(make_program_dp2, rank)
            ops = dist_main_prog.global_block().ops
            for op in ops:
                axes = op.desc.attr('axes')
                op_dist_attr = dist_context.get_op_dist_attr_for_program(op)
                assert op_dist_attr.impl_type == "slice"
                for out in op.output_arg_names:
                    var_dims_mapping = op_dist_attr.get_output_dims_mapping(out)

    def test_dist_slice_serial(self):
        dist_main_prog, dist_context = parallelizer(make_program_serial, 0)
        ops = dist_main_prog.global_block().ops
        for op in ops:
            op_dist_attr = dist_context.get_op_dist_attr_for_program(op)
            # We amend this impl_type after completion
            assert op_dist_attr.impl_type == "default"
            for out in op.output_arg_names:
                var_dims_mapping = op_dist_attr.get_output_dims_mapping(out)
                ref_dims_mapping = [-1 for i in range(len(var_dims_mapping))]
                assert ref_dims_mapping == ref_dims_mapping


if __name__ == "__main__":
    unittest.main()
