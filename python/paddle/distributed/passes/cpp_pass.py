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

from .pass_base import PassType, CPPPassWrapper, register_pass
from paddle.fluid.framework import core, _apply_pass as _apply_cpp_pass


@register_pass("fuse_elewise_add_act")
class FuseElementwiseAddActPass(CPPPassWrapper):

    def __init__(self):
        super(FuseElementwiseAddActPass, self).__init__()

    @property
    def cpp_name(self):
        return "fuse_elewise_add_act_pass"

    def _type(self):
        return PassType.FUSION_OPT


@register_pass("fuse_bn_act")
class FuseBatchNormActPass(CPPPassWrapper):

    def __init__(self):
        super(FuseBatchNormActPass, self).__init__()

    @property
    def cpp_name(self):
        return "fuse_bn_act_pass"

    def _type(self):
        return PassType.FUSION_OPT


@register_pass("fuse_bn_add_act")
class FuseBatchNormAddActPass(CPPPassWrapper):

    def __init__(self):
        super(FuseBatchNormAddActPass, self).__init__()

    @property
    def cpp_name(self):
        return "fuse_bn_add_act_pass"

    def _type(self):
        return PassType.FUSION_OPT


@register_pass("fuse_relu_depthwise_conv")
class FuseReluDepthwiseConvPass(CPPPassWrapper):

    def __init__(self):
        super(FuseReluDepthwiseConvPass, self).__init__()

    @property
    def cpp_name(self):
        return "fuse_relu_depthwise_conv_pass"

    def _type(self):
        return PassType.FUSION_OPT


@register_pass("fuse_optimizer")
class FuseOptimizerPass(CPPPassWrapper):

    def __init__(self):
        super(FuseOptimizerPass, self).__init__()

    @property
    def cpp_name(self):
        return [
            "fuse_adam_op_pass", "fuse_sgd_op_pass", "fuse_momentum_op_pass"
        ]

    def _type(self):
        return PassType.FUSION_OPT


@register_pass("inplace_addto_op")
class InplaceAddtoOpPass(CPPPassWrapper):

    def __init__(self):
        super(InplaceAddtoOpPass, self).__init__()

    @property
    def cpp_name(self):
        return "inplace_addto_op_pass"

    def _type(self):
        return PassType.CALC_OPT


@register_pass("build_cinn")
class BuildCINNPass(CPPPassWrapper):

    def __init__(self):
        super(BuildCINNPass, self).__init__()
        self.set_attr("allow_ops", [])
        self.set_attr("deny_ops", [])

    @property
    def cpp_name(self):
        return "build_cinn_pass"

    def _type(self):
        return PassType.CALC_OPT

    def _apply_single_impl(self, main_program, startup_program, context):
        allow_ops = ";".join(self.get_attr("allow_ops"))
        deny_ops = ";".join(self.get_attr("deny_ops"))

        assert 'FLAGS_allow_cinn_ops' in core.globals(
        ), "PaddlePaddle is not compiled with CINN support"
        old_allow_ops = core.globals()['FLAGS_allow_cinn_ops']
        old_deny_ops = core.globals()['FLAGS_deny_cinn_ops']
        try:
            core.globals()['FLAGS_allow_cinn_ops'] = allow_ops
            core.globals()['FLAGS_deny_cinn_ops'] = deny_ops
            _apply_cpp_pass(main_program, startup_program, self.cpp_name, {},
                            self.cpp_attr_types)
        finally:
            core.globals()['FLAGS_allow_cinn_ops'] = old_allow_ops
            core.globals()['FLAGS_deny_cinn_ops'] = old_deny_ops
