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
import copy

import paddle
import paddle.distributed.auto_parallel as auto
from paddle.distributed.auto_parallel.cluster import Cluster
from paddle.distributed.auto_parallel.operators.common import get_distributed_operator_impl_container

from paddle.fluid import program_guard
from paddle.fluid.backward import append_backward
from paddle.fluid.backward import append_backward

paddle.enable_static()


def parallelizer(program_func, rank):
    from paddle.distributed.auto_parallel.completion import Completer
    from paddle.distributed.auto_parallel.partitioner import Partitioner
    from paddle.distributed.auto_parallel.dist_context import DistributedContext

    main_program, startup_program, loss = program_func()

    # complete forward
    dist_context = DistributedContext()
    completer = Completer(dist_context)
    completer.complete_forward_annotation(main_program)
    dist_context.block_state.parse_forward_blocks(main_program)

    # generate backward and complete backward
    with paddle.static.program_guard(main_program, startup_program):
        params_grads = append_backward(
            loss, None, None, None, distop_context=dist_context.dist_op_context)
    completer.complete_backward_annotation(main_program)
    dist_context.block_state.parse_backward_blocks(main_program)

    optimizer = paddle.optimizer.SGD(learning_rate=0.001)
    # generate opt and complete opt
    with program_guard(main_program, startup_program):
        optimize_ops = copy.deepcopy(optimizer).apply_gradients(params_grads)

    completer.complete_update_annotation(main_program)

    return main_program, dist_context


class TestDistOpCost(unittest.TestCase):

    def test_dist_fill_constatnt_batch_size_like_op_cost(self):

        def make_program():
            main_program = paddle.static.Program()
            start_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, start_program):
                x = paddle.static.data(name='x', shape=[4, 8], dtype='float32')
                x.stop_gradient = True
                label = paddle.static.data(name="label",
                                           shape=[4, 1],
                                           dtype='float32')
                label.stop_gradient = True
                auto.shard_tensor(x,
                                  dist_attr={
                                      "process_mesh": auto.ProcessMesh([0, 1]),
                                      "dims_mapping": [0, -1]
                                  })
                tmp = paddle.fluid.layers.fill_constant_batch_size_like(
                    input=x, shape=[2, 8], value=1, dtype='float32')
                weight_attr = paddle.ParamAttr()
                linear = paddle.nn.Linear(8, 8, weight_attr=weight_attr)
                linear_out = linear(x)
                # default op with dp
                tmp = paddle.static.nn.layer_norm(linear_out)
                error_cost = paddle.nn.functional.square_error_cost(tmp, label)
                loss = paddle.mean(error_cost)
            return main_program, start_program, loss

        main_program, dist_context = parallelizer(make_program, 0)
        ops = main_program.global_block().ops
        cluster = Cluster()
        cluster.gen_default_config_cluster(device_count=2)
        for idx, op in enumerate(ops):
            if op.type != "matmul_v2" and op.type != "matmul_v2_grad" and op.type != "sgd":
                dist_op = dist_context.get_dist_op_for_program(op)
                op_dist_attr = dist_op.dist_attr
                processes = op_dist_attr.process_mesh.processes
                container = get_distributed_operator_impl_container(
                    op_dist_attr.impl_type)
                dist_impl = container.impls[op_dist_attr.impl_idx]
                dist_op_cost = dist_impl.calc_cost(op.attr('op_role'), dist_op,
                                                   dist_context, cluster)
                self.assertTrue(dist_op_cost)


if __name__ == "__main__":
    unittest.main()
