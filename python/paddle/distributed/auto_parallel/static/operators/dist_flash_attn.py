# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License

import logging

from ....utils.log_utils import get_logger

_logger = get_logger(logging.INFO)
from ...random import determinate_rng, is_enable_auto_rand_ctrl
from .common import (
    DistributedOperatorImplContainer,
    register_distributed_operator_impl,
    register_distributed_operator_impl_container,
)
from .dist_eltwise import DistributedDefaultImpl0, DistributedElementwiseImpl0


class DistributedFlashAttn(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super().__init__(op_type)


register_distributed_operator_impl_container(DistributedFlashAttn("flash_attn"))


# Dist FlashAttn with Random Control
class DistributedFlashAttnImpl0(DistributedElementwiseImpl0):
    def __init__(self, name):
        super().__init__(name)
        self._forward_implemented = True
        self._backward_implemented = True

    def is_input_compatible(self, dist_op):
        return True

    def is_output_compatible(self, dist_op):
        return True

    def is_auto_compatible(self, dist_op):
        return True

    @staticmethod
    def forward(ctx, *args, **kwargs):

        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.work_block
        startup_block = dist_op_context.startup_block
        src_op = dist_op_context.cur_src_op
        rank_id = dist_op_context.rank_id
        op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)

        if (
            is_enable_auto_rand_ctrl()
            and not op_dist_attr.is_recompute
            and rank_id in op_dist_attr.process_mesh.process_ids
        ):

            assert (
                op_dist_attr is not None
            ), f"forward op [{str(src_op)}] don't have dist attribute !"

            if (
                len(kwargs.get('fixed_seed_offset', [])) > 0
                or len(src_op.input("fixed_seed_offset")) > 0
            ):
                # TODO(kuizhiqing) recompute should go here
                pass
            else:
                # determinate rng
                q_var = main_block._var_recursive(kwargs['q'][0])
                k_var = main_block._var_recursive(kwargs['k'][0])
                q_dims_mapping = op_dist_attr.get_input_dims_mapping(q_var.name)
                k_dims_mapping = op_dist_attr.get_input_dims_mapping(k_var.name)
                process_mesh = op_dist_attr.process_mesh
                dims_mapping = q_dims_mapping[:3] + [q_dims_mapping[2]]

                rng_name = determinate_rng(rank_id, dims_mapping, process_mesh)
                assert rng_name is not None and rng_name != ""

                src_op._set_attr('rng_name', rng_name)

        DistributedDefaultImpl0.forward(ctx, *args, **kwargs)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        # dropout backward is deterministic by mask, and not need for random state control
        DistributedDefaultImpl0.backward(ctx, *args, **kwargs)


register_distributed_operator_impl(
    "flash_attn", DistributedFlashAttnImpl0("random_control")
)
