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

from paddle.distributed.auto_parallel.static.operators.common import (
    is_amp_flag_sync_op,
    is_data_parallel_reduce_op,
    is_global_norm_sync_op,
)
from paddle.distributed.auto_parallel.static.utils import (
    OpRole,
    insert_dependencies_for_vars,
)

from .auto_parallel_sharding import ShardingPass, _supported_optimizer_type
from .pass_base import PassBase, register_pass


def _sharding_pass_applied(pass_ctx):
    for applied_pass in pass_ctx.passes:
        if isinstance(applied_pass, ShardingPass):
            return True
    return False


# NOTE we add the "auto_parallel" prefix to the pass in order to
# indicate that this pass should obey some constrains by auto_parallel
# for example all ops and vars should has dist attr before and after pass
# should use dist op instead of custom comm op
@register_pass("auto_parallel_supplement_explicit_dependencies")
class AutoParalSupplementDepPass(PassBase):
    """
    Functional Concern.
    for strategies like amp & global norm, there is a collective communication to sync gradient inforation in every rank.
    after partition the gradients to each rank, the order of that collective communication is different in each rank
    and might cause hang problem in graph based random order executor. here supplement explicit dependencies for those cases.

    TODO Performance Concern.
    global collective will introduce global synchronization which forces the fast workers to wait for slow ones.
    therefore we should conduct this collective when all the ranks reach a same stage.
    BUT the depend API offered by executor could only ensure "conduct-not-before" but not "conduct-right-after".
    Some ranks might call the colletives first than other ranks while they still some local could be performed to wait for slow peers.
    IR Pass currently could not have the fully control of time the to perform these global collectives.
    """

    def __init__(self):
        super().__init__()
        self.set_attr("dist_context", None)

    def _check_self(self):
        if self.get_attr("dist_context") is None:
            return False

        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):

        # TODO general this pass for all case.
        if not _sharding_pass_applied(context):
            return

        self._dist_context = self.get_attr("dist_context", None)
        self.flags_sync_stream = "flags_sync_stream"
        main_block = main_program.global_block()
        startup_block = startup_program.global_block()

        # last dp grad communication
        last_dp_reduce_op_idx = -1
        last_dp_reduce_varname = None
        for idx, op in reversed(list(enumerate(main_block.ops))):
            if is_data_parallel_reduce_op(op):
                last_dp_reduce_op_idx = idx
                last_dp_reduce_varname = op.output_arg_names[0]
                break
        assert last_dp_reduce_op_idx > 0
        assert last_dp_reduce_varname is not None

        # analyze deps for amp & global norm
        deps_map = {}
        prior_varname = last_dp_reduce_varname
        for idx, op in enumerate(main_block.ops):
            if is_amp_flag_sync_op(op) or is_global_norm_sync_op(op):
                op_namescope = None
                if is_amp_flag_sync_op(op):
                    op_namescope = "amp_flag_sync_dep"
                    op.dist_attr.execution_stream = self.flags_sync_stream

                elif is_global_norm_sync_op(op):
                    op_namescope = "global_norm_sync_dep"
                deps_map[idx] = (prior_varname, op.input("X")[0], op_namescope)
                prior_varname = op.output("Out")[0]

        # analyze deps for check_finite_and_unscale
        # ensure it is performed after last backward computation, therefore reduce the
        # straggling of the amp-flag-sync
        first_check_op = True
        for idx, op in enumerate(main_block.ops):
            if op.type == "check_finite_and_unscale":
                if first_check_op:
                    last_backward_op = main_block.ops[idx - 1]
                    prior_varname = last_backward_op.output_arg_names[0]
                    first_check_op = False
                deps_map[idx] = (
                    prior_varname,
                    op.input("Scale")[0],
                    "check_finite_dep",
                )

        # analyze deps for optimizer
        # optimizers order should be fixed to allow broadcast to overlap with optimizer
        first_optimizer_op = True
        for idx, op in enumerate(main_block.ops):
            if op.type in _supported_optimizer_type:
                if first_optimizer_op:
                    first_optimizer_op = False
                else:
                    deps_map[idx] = (
                        prior_varname,
                        op.input("Param")[0],
                        "optimizer_order_dep",
                    )
                prior_varname = op.output("ParamOut")[0]

        # insert deps
        indice = sorted(deps_map.keys(), reverse=True)
        for idx in indice:
            prior_var = main_block.var(deps_map[idx][0])
            post_var = main_block.var(deps_map[idx][1])
            op_namescope = deps_map[idx][2]
            depend_op = insert_dependencies_for_vars(
                main_block,
                idx,
                prior_var,
                post_var,
                self._dist_context,
                OpRole.Optimize,
                is_recompute=False,
                sync=False,
                op_namescope=op_namescope,
            )

        main_block._sync_with_cpp()
