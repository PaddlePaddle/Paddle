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

<<<<<<< HEAD
from paddle.framework import _apply_pass as _apply_cpp_pass
from paddle.framework import core
from paddle.static import Executor

from .pass_base import CPPPassWrapper, PassType, register_pass
=======
from paddle.static import Executor
from .pass_base import PassType, CPPPassWrapper, register_pass
from paddle.fluid.framework import core, _apply_pass as _apply_cpp_pass
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


@register_pass("fuse_elewise_add_act")
class FuseElementwiseAddActPass(CPPPassWrapper):
<<<<<<< HEAD
    def __init__(self):
        super().__init__()
=======

    def __init__(self):
        super(FuseElementwiseAddActPass, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    @property
    def cpp_name(self):
        return "fuse_elewise_add_act_pass"

    def _type(self):
        return PassType.FUSION_OPT


@register_pass("fuse_bn_act")
class FuseBatchNormActPass(CPPPassWrapper):
<<<<<<< HEAD
    def __init__(self):
        super().__init__()
=======

    def __init__(self):
        super(FuseBatchNormActPass, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    @property
    def cpp_name(self):
        return "fuse_bn_act_pass"

    def _type(self):
        return PassType.FUSION_OPT


@register_pass("fuse_bn_add_act")
class FuseBatchNormAddActPass(CPPPassWrapper):
<<<<<<< HEAD
    def __init__(self):
        super().__init__()
=======

    def __init__(self):
        super(FuseBatchNormAddActPass, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    @property
    def cpp_name(self):
        return "fuse_bn_add_act_pass"

    def _type(self):
        return PassType.FUSION_OPT


@register_pass("fuse_relu_depthwise_conv")
class FuseReluDepthwiseConvPass(CPPPassWrapper):
<<<<<<< HEAD
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


@register_pass("fuse_gemm_epilogue")
class FuseGemmEpiloguePass(CPPPassWrapper):
    def __init__(self):
        super().__init__()

    @property
    def cpp_name(self):
        return "fuse_gemm_epilogue_pass"
=======

    def __init__(self):
        super(FuseReluDepthwiseConvPass, self).__init__()

    @property
    def cpp_name(self):
        return "fuse_relu_depthwise_conv_pass"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def _type(self):
        return PassType.FUSION_OPT


@register_pass("fuse_optimizer")
class FuseOptimizerPass(CPPPassWrapper):
<<<<<<< HEAD
    def __init__(self):
        super().__init__()
=======

    def __init__(self):
        super(FuseOptimizerPass, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    @property
    def cpp_name(self):
        return [
<<<<<<< HEAD
            "fuse_adam_op_pass",
            "fuse_sgd_op_pass",
            "fuse_momentum_op_pass",
=======
            "fuse_adam_op_pass", "fuse_sgd_op_pass", "fuse_momentum_op_pass"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        ]

    def _type(self):
        return PassType.FUSION_OPT


@register_pass("inplace_addto_op")
class InplaceAddtoOpPass(CPPPassWrapper):
<<<<<<< HEAD
    def __init__(self):
        super().__init__()
=======

    def __init__(self):
        super(InplaceAddtoOpPass, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    @property
    def cpp_name(self):
        return "inplace_addto_op_pass"

    def _type(self):
        return PassType.CALC_OPT


def _set_cinn_op_flag(flag_name, extra_ops):
    values = core.globals()[flag_name]
    values = [v.strip() for v in values.split(";") if v.strip()]
    values.extend(extra_ops)
    core.globals()[flag_name] = ";".join(values)


@register_pass("build_cinn")
class BuildCINNPass(CPPPassWrapper):
<<<<<<< HEAD
    def __init__(self):
        super().__init__()
=======

    def __init__(self):
        super(BuildCINNPass, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.set_attr("allow_ops", [])
        self.set_attr("deny_ops", [])

    @property
    def cpp_name(self):
        return "build_cinn_pass"

    def _type(self):
        return PassType.CALC_OPT

    def _apply_single_impl(self, main_program, startup_program, context):

<<<<<<< HEAD
        assert (
            'FLAGS_allow_cinn_ops' in core.globals()
=======
        assert 'FLAGS_allow_cinn_ops' in core.globals(
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        ), "PaddlePaddle is not compiled with CINN support"
        old_allow_ops = core.globals()['FLAGS_allow_cinn_ops']
        old_deny_ops = core.globals()['FLAGS_deny_cinn_ops']
        try:
<<<<<<< HEAD
            _set_cinn_op_flag(
                'FLAGS_allow_cinn_ops', self.get_attr("allow_ops")
            )
=======
            _set_cinn_op_flag('FLAGS_allow_cinn_ops',
                              self.get_attr("allow_ops"))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            _set_cinn_op_flag('FLAGS_deny_cinn_ops', self.get_attr("deny_ops"))

            feed = self.get_attr('feed', [])
            fetch_list = self.get_attr('fetch_list', [])
            prune_program = self.get_attr('prune_program', True)

            if prune_program:
                tmp_main_program = Executor._prune_program(
<<<<<<< HEAD
                    main_program, feed, fetch_list, []
                )

                tmp_main_program = Executor._add_fetch_ops(
                    tmp_main_program, fetch_list, 'fetch'
                )
=======
                    main_program, feed, fetch_list, [])

                tmp_main_program = Executor._add_fetch_ops(
                    tmp_main_program, fetch_list, 'fetch')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            else:

                tmp_main_program = Executor._add_fetch_ops(
<<<<<<< HEAD
                    main_program, fetch_list, 'fetch'
                )

            _apply_cpp_pass(
                tmp_main_program,
                startup_program,
                self.cpp_name,
                {},
                self.cpp_attr_types,
            )
=======
                    main_program, fetch_list, 'fetch')

            _apply_cpp_pass(tmp_main_program, startup_program, self.cpp_name,
                            {}, self.cpp_attr_types)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            tmp_main_program = Executor._remove_fetch_ops(tmp_main_program)

            tmp_main_program = core.ProgramDesc(tmp_main_program.desc)

            main_program._rebuild_from_desc(tmp_main_program)

        finally:
            core.globals()['FLAGS_allow_cinn_ops'] = old_allow_ops
            core.globals()['FLAGS_deny_cinn_ops'] = old_deny_ops
