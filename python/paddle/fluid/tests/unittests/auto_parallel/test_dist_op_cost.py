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
import paddle.distributed.auto_parallel as auto
from paddle.distributed.auto_parallel.operators.common import get_distributed_operator_impl_container

from paddle.fluid import program_guard
from paddle.fluid.backward import append_backward
from paddle.distributed.auto_parallel.utils import print_program_with_dist_attr

paddle.enable_static()


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


class TestDistOpCost(unittest.TestCase):

    def test_dist_fill_constatnt_batch_size_like_op_cost(self):

        def make_program():
            main_program = paddle.fluid.Program()
            start_program = paddle.fluid.Program()
            with paddle.static.program_guard(main_program, start_program):
                x = paddle.static.data(name='x', shape=[4, 8], dtype='float32')
                x.stop_gradient = False
                label = paddle.static.data(name="label",
                                           shape=[batch_size, 1],
                                           dtype='float32')
                auto.shard_tensor(x,
                                  dist_attr={
                                      "process_mesh": auto.ProcessMesh([0, 1]),
                                      "dims_mapping": [0, -1]
                                  })
                tmp = paddle.fluid.layers.fill_constant_batch_size_like(
                    input=x, shape=[2, 8], value=0, dtype='float32')
                error_cost = paddle.nn.functional.square_error_cost(tmp, label)
                loss = paddle.mean(error_cost)
            optimizer = paddle.fluid.optimizer.AdamOptimizer(
                learning_rate=0.00001,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-08,
                grad_clip=None)
            _, _, main_program, start_program = optimizer.minimize(
                loss, start_program)

            return main_program, start_program

        dist_main_prog, dist_context = parallelizer(make_program, 0)
        ops = dist_main_prog.global_block().ops
        for idx, op in enumerate(ops):
            if op.type == "fill_constant_batch_size_like" or op.type == "fill_constant_batch_size_like_grad":
                dist_op = dist_context.get_dist_op_for_program(op)
                op_dist_attr = dist_op.dist_attr
                processes = op_dist_attr.process_mesh.processes
                container = get_distributed_operator_impl_container(
                    op_dist_attr.impl_type)
                dist_impl = container.impls[op_dist_attr.impl_idx]
                dist_op_cost = dist_impl.calc_cost(op.attr('op_role'), dist_op,
                                                   dist_context, self.cluster)
                print(dist_op_cost)
                self.assertTrue(dist_op_cost)


if __name__ == "__main__":
    unittest.main()
