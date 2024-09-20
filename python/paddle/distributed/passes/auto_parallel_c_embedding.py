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

import paddle
import paddle.distributed as dist
from paddle.distributed.fleet.meta_optimizers.common import OpRole

from .pass_base import PassBase, register_pass


@register_pass("auto_parallel_c_embedding_pass")
class AutoParallelCEmbeddingPass(PassBase):
    def __init__(self):
        super().__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        block = main_program.global_block()
        ops = block.ops
        for i, op in enumerate(ops):
            if op.name() == 'pd_op.embedding':
                # replace embedding with c_embedding
                paddle.pir.set_insertion_point(op)
                num_embeddings = op.operand(1).source().type().shape[0]
                world_size = paddle.distributed.get_world_size()
                rank = paddle.distributed.get_rank()
                per_part_size = num_embeddings // world_size
                vocab_start_index = rank * per_part_size
                t_op = paddle._C_ops.c_embedding(
                    op.operand(1).source(),
                    op.operand(0).source(),
                    vocab_start_index,
                    num_embeddings,
                )
                t_op.get_defining_op().op_role = int(OpRole.Optimize)
                new_op = t_op.get_defining_op()
                op.result(0).replace_all_uses_with(t_op)
                op.erase()

  

                # # input0 weight
                placements_input0 = new_op.operand(0).source().placements
                dim_map_input0, partial_status_input0 = (
                    dist.auto_parallel.placement_type.to_dim_map(
                        placements_input0, len(placements_input0)
                    )
                )
                dim_map_input0 = [1, -1]
                dist_attr_input0 = (
                    paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                        new_op.operand(0).source().process_mesh,
                        dim_map_input0,
                        partial_status_input0,
                    )
                )
                dist_type_input0 = paddle.base.libpaddle.pir.cvt_to_dist_type(
                    new_op.operand(0).source().type(), dist_attr_input0
                )
                new_op.operand(0).source().set_type(dist_type_input0)

                # # input1 x
                placements_input1 = new_op.operand(1).source().placements
                dim_map_input1, partial_status_input1 = (
                    dist.auto_parallel.placement_type.to_dim_map(
                        placements_input1, len(placements_input1)
                    )
                )
                dist_attr_input1 = (
                    paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                        new_op.operand(1).source().process_mesh,
                        dim_map_input1,
                        partial_status_input1,
                    )
                )

                # output
                placements_out0 = new_op.results()[0].placements
                dim_map_out0, partial_status_out0 = (
                    dist.auto_parallel.placement_type.to_dim_map(
                        placements_out0, len(placements_out0)
                    )
                )
                dim_map_out0 = [-1, -1, -1]
                partial_status_out0 = {1: paddle.base.core.ReduceType.kRedSum}
                dist_attr_out0 = (
                    paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                        new_op.results()[0].process_mesh,
                        dim_map_out0,
                        partial_status_out0,
                    )
                )
                dist_type_out0 = paddle.base.libpaddle.pir.cvt_to_dist_type(
                    new_op.results()[0].type(), dist_attr_out0
                )
                new_op.results()[0].set_type(dist_type_out0)

                new_op.dist_attr = (
                    paddle.base.libpaddle.pir.create_op_dist_attribute(
                        new_op.operand(0).source().process_mesh,
                        [dist_attr_input0, dist_attr_input1],
                        [dist_attr_out0],
                    )
                )

                # replace allgather with allreduce
                for op in new_op.results()[0].all_used_ops():
                    placements_in = op.operand(0).source().placements
                    dim_map_in, partial_status_in = (
                        dist.auto_parallel.placement_type.to_dim_map(
                            placements_in, len(placements_in)
                        )
                    )
                    dim_map_in = [-1, -1, -1]
                    # partial_status_in = {1: paddle.base.core.ReduceType.kRedSum}
                    dist_attr_in = (
                        paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                            op.operand(0).source().process_mesh,
                            dim_map_in,
                            partial_status_in,
                        )
                    )
                    dist_type_in = paddle.base.libpaddle.pir.cvt_to_dist_type(
                        op.operand(0).source().type(), dist_attr_in
                    )
                    op.operand(0).source().set_type(dist_type_in)

                    placements_out = op.results()[0].placements
                    dim_map_out, partial_status_out = (
                        dist.auto_parallel.placement_type.to_dim_map(
                            placements_out, len(placements_out)
                        )
                    )
                    print("dim_map_out:", dim_map_out)
                    # dim_map_out = [0, -1, -1]
                    dim_map_out = [-1, -1, -1] # hack dome
                    print("partial_status_out:", partial_status_out)
                    dist_attr_out = (
                        paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                            op.results()[0].process_mesh,
                            dim_map_out,
                            partial_status_out,
                        )
                    )
                    dist_type_out = paddle.base.libpaddle.pir.cvt_to_dist_type(
                        op.results()[0].type(), dist_attr_out
                    )
                    op.results()[0].set_type(dist_type_out)

                    op.dist_attr = (
                        paddle.base.libpaddle.pir.create_op_dist_attribute(
                            op.operand(0).source().process_mesh,
                            [dist_attr_in],
                            [dist_attr_out],
                        )
                    )

            # elif op.name() == 'pd_op.embedding_grad1':

            #     # replace embedding_grad with c_embedding_grad
            #     paddle.pir.set_insertion_point(op)
            #     num_embeddings = op.operand(1).source().type().shape[0]
            #     world_size = paddle.distributed.get_world_size()
            #     rank = paddle.distributed.get_rank()
            #     per_part_size = num_embeddings // world_size
            #     vocab_start_index = rank * per_part_size
            #     t_op = paddle._C_ops.c_embedding_grad(
            #         op.operand(1).source(),
            #         op.operand(0).source(),
            #         op.operand(2).source(),
            #         vocab_start_index,
            #         num_embeddings,
            #     )
            #     t_op.get_defining_op().op_role = int(OpRole.Optimize)
            #     new_op = t_op.get_defining_op()
            #     op.result(0).replace_all_uses_with(t_op)
            #     op.erase()

            #     # output
            #     placements_out0 = new_op.results()[0].placements
            #     dim_map_out0, partial_status_out0 = (
            #         dist.auto_parallel.placement_type.to_dim_map(
            #             placements_out0, len(placements_out0)
            #         )
            #     )
            #     dim_map_out0 = [1, -1]
            #     dist_attr_out0 = (
            #         paddle.base.libpaddle.pir.create_tensor_dist_attribute(
            #             new_op.results()[0].process_mesh,
            #             dim_map_out0,
            #             partial_status_out0,
            #         )
            #     )
            #     dist_type_out0 = paddle.base.libpaddle.pir.cvt_to_dist_type(
            #         new_op.results()[0].type(), dist_attr_out0
            #     )
            #     new_op.results()[0].set_type(dist_type_out0)

            #     # input0 weight 这里是不是应该是use replcae
            #     placements_input0 = new_op.operand(0).source().placements
            #     dim_map_input0, partial_status_input0 = (
            #         dist.auto_parallel.placement_type.to_dim_map(
            #             placements_input0, len(placements_input0)
            #         )
            #     )
            #     dim_map_input0 = [1, -1]
            #     dist_attr_input0 = (
            #         paddle.base.libpaddle.pir.create_tensor_dist_attribute(
            #             new_op.operand(0).source().process_mesh,
            #             dim_map_input0,
            #             partial_status_input0,
            #         )
            #     )
            #     dist_type_input0 = paddle.base.libpaddle.pir.cvt_to_dist_type(
            #         new_op.operand(0).source().type(), dist_attr_input0
            #     )
            #     new_op.operand(0).source().set_type(dist_type_input0)

            #     # input1 x
            #     placements_input1 = new_op.operand(1).source().placements
            #     dim_map_input1, partial_status_input1 = (
            #         dist.auto_parallel.placement_type.to_dim_map(
            #             placements_input1, len(placements_input1)
            #         )
            #     )
            #     dist_attr_input1 = (
            #         paddle.base.libpaddle.pir.create_tensor_dist_attribute(
            #             new_op.operand(1).source().process_mesh,
            #             dim_map_input1,
            #             partial_status_input1,
            #         )
            #     )

            #     # input2
            #     placements_input2 = new_op.operand(2).source().placements
            #     dim_map_input2, partial_status_input2 = (
            #         dist.auto_parallel.placement_type.to_dim_map(
            #             placements_input2, len(placements_input2)
            #         )
            #     )
            #     dim_map_input2 = [-1, -1, -1]
            #     dist_attr_input2 = (
            #         paddle.base.libpaddle.pir.create_tensor_dist_attribute(
            #             new_op.operand(2).source().process_mesh,
            #             dim_map_input2,
            #             {1: paddle.base.core.ReduceType.kRedSum},
            #         )
            #     )

            #     new_op.dist_attr = (
            #         paddle.base.libpaddle.pir.create_op_dist_attribute(
            #             new_op.operand(0).source().process_mesh,
            #             [dist_attr_input0, dist_attr_input1, dist_attr_input2],
            #             [dist_attr_out0],
            #         )
            #     )

            #     # replace allgather with allreduce
            #     # new_op.operand(0).get_defining_op()
            #     print(
            #         "## lzx ## get_defining_op;",
            #         op.operand(2).source().get_defining_op(),
            #     )
            #     op = op.operand(2).source().get_defining_op()

            #     placements_in = op.operand(0).source().placements
            #     dim_map_in, partial_status_in = (
            #         dist.auto_parallel.placement_type.to_dim_map(
            #             placements_in, len(placements_in)
            #         )
            #     )
            #     # dim_map_in = [-1, -1, -1]
            #     dist_attr_in = (
            #         paddle.base.libpaddle.pir.create_tensor_dist_attribute(
            #             op.operand(0).source().process_mesh,
            #             dim_map_in,
            #             partial_status_in,
            #         )
            #     )

            #     placements_out = op.results()[0].placements
            #     dim_map_out, partial_status_out = (
            #         dist.auto_parallel.placement_type.to_dim_map(
            #             placements_out, len(placements_out)
            #         )
            #     )
            #     dim_map_out = [-1, -1, -1]
            #     dist_attr_out = (
            #         paddle.base.libpaddle.pir.create_tensor_dist_attribute(
            #             op.results()[0].process_mesh,
            #             dim_map_out,
            #             {1: paddle.base.core.ReduceType.kRedSum},
            #         )
            #     )

            #     op.dist_attr = (
            #         paddle.base.libpaddle.pir.create_op_dist_attribute(
            #             op.operand(0).source().process_mesh,
            #             [dist_attr_in],
            #             [dist_attr_out],
            #         )
            #     )
            #     dist_type_out = paddle.base.libpaddle.pir.cvt_to_dist_type(
            #         new_op.results()[0].type(), dist_attr_out0
            #     )
            #     op.results()[0].set_type(dist_type_out)
