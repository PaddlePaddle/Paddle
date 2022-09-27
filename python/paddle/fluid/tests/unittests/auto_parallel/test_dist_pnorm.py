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
        x = paddle.static.data(name='x', shape=[4, 5, 6], dtype='float32')
        x.stop_gradient = False
        auto.shard_tensor(x, auto.ProcessMesh([0, 1], dim_names=["x"]),
                          ["x", None, None])
        tmp_0 = paddle.norm(x, p=2)
    return main_program, start_program, tmp_0


def make_program_serial():
    main_program = paddle.fluid.Program()
    start_program = paddle.fluid.Program()
    with paddle.static.program_guard(main_program, start_program):
        x = paddle.static.data(name='x', shape=[4, 5, 6], dtype='float32')
        x.stop_gradient = False
        auto.shard_tensor(x, auto.ProcessMesh([0], dim_names=["x"]),
                          [None, None, None])
        tmp_0 = paddle.norm(x, p=2)
    return main_program, start_program, tmp_0


def parallelizer(program_func, rank):
    from paddle.distributed.auto_parallel.completion import Completer
    from paddle.distributed.auto_parallel.partitioner import Partitioner
    from paddle.distributed.auto_parallel.dist_context import DistributedContext

    main_program, start_program, loss = program_func()

    dist_context = DistributedContext()
    completer = Completer(dist_context)
    completer.complete_forward_annotation(main_program)
    dist_context.block_state.parse_forward_blocks(main_program)

    with program_guard(main_program, start_program):
        params_grads = append_backward(
            loss, distop_context=dist_context.dist_op_context)
    completer.complete_backward_annotation(main_program)

    dist_context.block_state.parse_backward_blocks(main_program)
    partitioner = Partitioner(dist_context, rank)
    dist_main_prog, _, _ = partitioner.partition(main_program, start_program,
                                                 [])

    return dist_main_prog, dist_context


class TestDistPNorm(unittest.TestCase):

    def test_dist_pnorm_dp2(self):

        for rank in range(2):
            dist_main_prog, dist_context = parallelizer(make_program_dp2, rank)
            ops = dist_main_prog.global_block().ops
            op_types = []
            for op in ops:
                op_types.append(op.type)
                op_dist_attr = dist_context.get_op_dist_attr_for_program(op)
                if op.type == "p_norm":
                    assert op_dist_attr.impl_type == "p_norm"
                if op.type in ["p_norm", "p_norm_grad"]:
                    for input_attr in op_dist_attr.inputs_dist_attrs.values():
                        assert set(input_attr.dims_mapping) == set([-1])
                    for output_attr in op_dist_attr.outputs_dist_attrs.values():
                        assert set(output_attr.dims_mapping) == set([-1])
                if op.type == 'c_allgather':
                    for input_attr in op_dist_attr.inputs_dist_attrs.values():
                        assert input_attr.dims_mapping[0] == 0
                        assert set(input_attr.dims_mapping[1:]) == set([-1])
                    for output_attr in op_dist_attr.outputs_dist_attrs.values():
                        assert set(output_attr.dims_mapping) == set([-1])
                if op.type == 'slice':
                    for input_attr in op_dist_attr.inputs_dist_attrs.values():
                        assert set(input_attr.dims_mapping) == set([-1])
                    for output_attr in op_dist_attr.outputs_dist_attrs.values():
                        assert output_attr.dims_mapping[0] == 0
                        assert set(output_attr.dims_mapping[1:]) == set([-1])
            assert op_types == [
                "c_allgather", "p_norm", "fill_constant", "p_norm_grad", "slice"
            ]

    def test_dist_pnorm_serial(self):
        dist_main_prog, dist_context = parallelizer(make_program_serial, 0)
        ops = dist_main_prog.global_block().ops
        for op in ops:
            op_dist_attr = dist_context.get_op_dist_attr_for_program(op)
            assert op_dist_attr.impl_type == "default"


if __name__ == "__main__":
    unittest.main()
