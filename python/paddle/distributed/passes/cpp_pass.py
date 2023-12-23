# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.framework import (
    _apply_pass as _apply_cpp_pass,
    core,
)
from paddle.static import Executor

from .pass_base import CPPPassWrapper, PassType, register_pass


@register_pass("fuse_elewise_add_act")
class FuseElementwiseAddActPass(CPPPassWrapper):
    def __init__(self):
        super().__init__()

    @property
    def cpp_name(self):
        return "fuse_elewise_add_act_pass"

    def _type(self):
        return PassType.FUSION_OPT


@register_pass("fuse_bn_act")
class FuseBatchNormActPass(CPPPassWrapper):
    def __init__(self):
        super().__init__()

    @property
    def cpp_name(self):
        return "fuse_bn_act_pass"

    def _type(self):
        return PassType.FUSION_OPT


@register_pass("fuse_bn_add_act")
class FuseBatchNormAddActPass(CPPPassWrapper):
    def __init__(self):
        super().__init__()

    @property
    def cpp_name(self):
        return "fuse_bn_add_act_pass"

    def _type(self):
        return PassType.FUSION_OPT


@register_pass("fuse_relu_depthwise_conv")
class FuseReluDepthwiseConvPass(CPPPassWrapper):
    def __init__(self):
        super().__init__()

    @property
    def cpp_name(self):
        return "fuse_relu_depthwise_conv_pass"

    def _type(self):
        return PassType.FUSION_OPT


@register_pass("fused_attention")
class FusedAttentionPass(CPPPassWrapper):
    def __init__(self):
        super().__init__()

    @property
    def cpp_name(self):
        return "fused_attention_pass"

    def _type(self):
        return PassType.FUSION_OPT


@register_pass("fused_feedforward")
class FusedFeedforwardPass(CPPPassWrapper):
    def __init__(self):
        super().__init__()

    @property
    def cpp_name(self):
        return "fused_feedforward_pass"

    def _type(self):
        return PassType.FUSION_OPT


@register_pass("fuse_gemm_epilogue")
class FuseGemmEpiloguePass(CPPPassWrapper):
    def __init__(self):
        super().__init__()

    @property
    def cpp_name(self):
        return "fuse_gemm_epilogue_pass"

    def _type(self):
        return PassType.FUSION_OPT


@register_pass("fuse_adamw")
class FuseAdamWPass(CPPPassWrapper):
    def __init__(self):
        super().__init__()

    @property
    def cpp_name(self):
        return "fuse_adamw_op_pass"

    def _type(self):
        return PassType.FUSION_OPT


@register_pass("fuse_dot_product_attention")
class FuseDotProductAttentionPass(CPPPassWrapper):
    def __init__(self):
        super().__init__()

    @property
    def cpp_name(self):
        return "fuse_dot_product_attention_pass"

    def _type(self):
        return PassType.FUSION_OPT


@register_pass("fuse_optimizer")
class FuseOptimizerPass(CPPPassWrapper):
    def __init__(self):
        super().__init__()

    @property
    def cpp_name(self):
        return [
            "fuse_adam_op_pass",
            "fuse_sgd_op_pass",
            "fuse_momentum_op_pass",
        ]

    def _type(self):
        return PassType.FUSION_OPT


@register_pass("inplace_addto_op")
class InplaceAddtoOpPass(CPPPassWrapper):
    def __init__(self):
        super().__init__()

    @property
    def cpp_name(self):
        return "inplace_addto_op_pass"

    def _type(self):
        return PassType.CALC_OPT


@register_pass("fuse_resunit")
class FuseResUnitPass(CPPPassWrapper):
    def __init__(self):
        super().__init__()

    @property
    def cpp_name(self):
        return "fuse_resunit_pass"

    def _type(self):
        return PassType.FUSION_OPT


def _set_cinn_op_flag(flag_name, extra_ops):
    values = core.globals()[flag_name]
    values = [v.strip() for v in values.split(";") if v.strip()]
    values.extend(extra_ops)
    core.globals()[flag_name] = ";".join(values)


@register_pass("build_cinn")
class BuildCINNPass(CPPPassWrapper):
    def __init__(self):
        super().__init__()
        self.set_attr("allow_ops", [])
        self.set_attr("deny_ops", [])

    @property
    def cpp_name(self):
        return "build_cinn_pass"

    def _type(self):
        return PassType.CALC_OPT

    def _apply_single_impl(self, main_program, startup_program, context):
        assert (
            'FLAGS_allow_cinn_ops' in core.globals()
        ), "PaddlePaddle is not compiled with CINN support"
        old_allow_ops = core.globals()['FLAGS_allow_cinn_ops']
        old_deny_ops = core.globals()['FLAGS_deny_cinn_ops']
        try:
            _set_cinn_op_flag(
                'FLAGS_allow_cinn_ops', self.get_attr("allow_ops")
            )
            _set_cinn_op_flag('FLAGS_deny_cinn_ops', self.get_attr("deny_ops"))

            feed = self.get_attr('feed', [])
            fetch_list = self.get_attr('fetch_list', [])
            prune_program = self.get_attr('prune_program', True)

            if prune_program:
                tmp_main_program = Executor._prune_program(
                    main_program, feed, fetch_list, []
                )

                tmp_main_program = Executor._add_fetch_ops(
                    tmp_main_program, fetch_list, 'fetch'
                )

            else:
                tmp_main_program = Executor._add_fetch_ops(
                    main_program, fetch_list, 'fetch'
                )

            _apply_cpp_pass(
                tmp_main_program,
                startup_program,
                self.cpp_name,
                {},
                self.cpp_attr_types,
            )

            tmp_main_program = Executor._remove_fetch_ops(tmp_main_program)

            tmp_main_program = core.ProgramDesc(tmp_main_program.desc)

            main_program._rebuild_from_desc(tmp_main_program)

        finally:
            core.globals()['FLAGS_allow_cinn_ops'] = old_allow_ops
            core.globals()['FLAGS_deny_cinn_ops'] = old_deny_ops
