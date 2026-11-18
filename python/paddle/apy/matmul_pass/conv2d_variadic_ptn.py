# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import abstract_drr
import access_topo_drr  # noqa: F401
import ap
import compile_command_util
import conv2d_variadic_tpl
import epilogue_access_topo_simplify
import ir_tools
import kernel_arg_id_util
import kernel_arg_translator_util  # noqa: F401
import low_level_ir_code_gen_ctx_util  # noqa: F401
import op_conversion_drr_pass  # noqa: F401
import pir  # noqa: F401
import umprime  # noqa: F401
import variadic_mixin


class Conv2dEpilogueFusion(variadic_mixin.VariadicEpilogueFusionBase):
    def source_pattern(self, o, t):
        in_num = self.number_of_inputs()
        out_num = self.number_of_outputs()
        o.conv2d_op = o.ap_native_op("pd_op.conv2d")
        o.conv2d_op([t.input0, t.input1], [t.conv2d_out])
        o.trivial_op = o.ap_trivial_fusion_op()
        o.trivial_op(
            [
                t.conv2d_out,
                *ap.map(
                    lambda index: getattr(t, f"input{index + 2}"),
                    range(in_num - 2),
                ),
            ],
            ap.map(lambda index: getattr(t, f"output{index}"), range(out_num)),
        )

    def result_pattern(self, o, t):
        in_num = self.number_of_inputs()
        out_num = self.number_of_outputs()
        o.fustion_op = o.ap_pattern_fusion_op(self.code_gen)
        o.fustion_op(
            ap.map(lambda index: getattr(t, f"input{index}"), range(in_num)),
            ap.map(lambda index: getattr(t, f"output{index}"), range(out_num)),
        )

    def constraint(self, o, t):
        return self._is_supported_by_device() and (
            self._is_supported_by_backend(o, t)
            and self._epilogue_can_fuse(o, t)
        )

    def _is_supported_by_device(self):
        compile_command_generator = (
            compile_command_util.CompileCommandGenerator()
        )
        return compile_command_generator.supports("conv2d")

    def _is_supported_by_backend(self, o, t):
        # Limitations of the cutlass conv2d backend:
        # - groups == 1: grouped conv needs a different cutlass kernel (GroupMode).
        # - NHWC: cutlass conv fprop only has TensorNHWC and
        #   TensorNCxHWx<Interleave> iterators, no plain NCHW.
        # - EXPLICIT and symmetric padding: `paddings` is forwarded as-is while
        #   cutlass takes a single pad value per spatial axis, and SAME / VALID
        #   derive the effective padding at runtime.
        # - KRSC filter, see `_is_krsc_filter`.
        # - static filter channels, see `_has_static_filter_channels`.
        paddings = self._get_int_list_attr(o.conv2d_op.paddings)
        symmetric_padding = (
            True
            if len(paddings) == 2
            else (paddings[0] == paddings[1] and paddings[2] == paddings[3])
        )
        supported_attrs = (self._get_int_attr(o.conv2d_op.groups) == 1) and (
            o.conv2d_op.data_format.match(a_str=lambda x: x) == "NHWC"
        )
        explicit_padding = (
            o.conv2d_op.padding_algorithm.match(a_str=lambda x: x) == "EXPLICIT"
        ) and symmetric_padding
        supported_conv_attrs = supported_attrs and explicit_padding
        return supported_conv_attrs and (
            self._is_krsc_filter(t) and self._has_static_filter_channels(t)
        )

    def _is_krsc_filter(self, t):
        # cutlass reads the filter as KRSC (`layout::TensorNHWC` on a filter
        # means "C innermost"), while an untransposed paddle filter is KCRS.
        # The two layouts store C in different slots, and the filter shape
        # alone cannot always tell them apart: a native KCRS filter whose S
        # equals C has exactly the same shape as a KRSC one, so matching on
        # the innermost dim alone would silently read the same buffer the
        # wrong way.
        input_shape = t.input0.symbolic_shape_to_list()
        weight_shape = t.input1.symbolic_shape_to_list()
        channels = input_shape[3]
        return (weight_shape[3] == channels) and (weight_shape[1] != channels)

    def _has_static_filter_channels(self, t):
        weight_shape = t.input1.symbolic_shape_to_list()
        is_static_c = weight_shape[3].match(
            int64=lambda channels: True, _=lambda: False
        )
        is_static_k = weight_shape[0].match(
            int64=lambda channels: True, _=lambda: False
        )
        return is_static_c and is_static_k

    def _epilogue_can_fuse(self, o, t):
        program = ir_tools.copy_fused_ops_to_program(
            o.trivial_op, tensor_match_ctx=t
        )
        program = epilogue_access_topo_simplify.simplify_epilogue_program(
            program,
            anchor_data_op_name="conv2d_out",
            number_of_inputs=self.number_of_inputs(),
            number_of_outputs=self.number_of_outputs(),
        )
        return program.empty()

    def _anchor_data_op_name(self):
        return "conv2d_out"

    def _anchor_input_count(self):
        return 2

    def _anchor_iter_var_names(self):
        return conv2d_variadic_tpl.get_anchor_iter_var_names()

    def _make_kernel_arg_translator(self):
        return conv2d_variadic_tpl.make_kernel_arg_translator()

    def _get_int_attr(self, attr):
        return attr.match(a_i32=lambda x: int(x))

    def _get_int_list_attr(self, attr):
        # `int[]` of pd_op.conv2d is ArrayAttribute<Int32Attribute>.
        return attr.match(
            a_array=lambda values: ap.map(self._get_int_attr, values)
        )

    def code_gen(self, ctx, o, t):
        program_translator = self._get_program_translator(ctx, o, t)
        mut_kernel_arg_id_registry = kernel_arg_id_util.KernelArgIdNameRegistry(
            code_gen_ctx=ctx, tensor_match_ctx=t, name_prefix=""
        )

        template_module = conv2d_variadic_tpl.Conv2dVariadicTemplate(
            program_translator=program_translator,
            mut_kernel_arg_id_registry=mut_kernel_arg_id_registry,
        )

        def get_symbolic_shape_args_list(sym_dim):
            return ctx.dim_expr_kernel_arg_id(sym_dim)

        input0_shape_kargs = ap.map(
            get_symbolic_shape_args_list, t.input0.symbolic_shape_to_list()
        )
        input1_shape_kargs = ap.map(
            get_symbolic_shape_args_list, t.input1.symbolic_shape_to_list()
        )
        return template_module.compile(
            input0_karg=ctx.in_tensor_data_ptr_kernel_arg_id(t.input0),
            input1_karg=ctx.in_tensor_data_ptr_kernel_arg_id(t.input1),
            output_karg=ctx.out_tensor_data_ptr_kernel_arg_id(t.output0),
            input0_shape_kargs=input0_shape_kargs,
            input1_shape_kargs=input1_shape_kargs,
            strides=self._get_int_list_attr(o.conv2d_op.strides),
            paddings=self._get_int_list_attr(o.conv2d_op.paddings),
            dilations=self._get_int_list_attr(o.conv2d_op.dilations),
            groups=self._get_int_attr(o.conv2d_op.groups),
            data_format=o.conv2d_op.data_format.match(a_str=lambda x: x),
        )


def register_conv2d_epilogue_class(base_class, max_num_inputs, max_num_outputs):
    def register_conv2d_drr_class(num_inputs, num_outputs):
        abstract_drr.register_drr_pass(
            f"conv2d_epilogue_in{num_inputs}_out{num_outputs}_fusion", nice=0
        )(
            variadic_mixin.get_mixin_class(
                base_class,
                "Conv2dEpilogueFusion",
                num_inputs,
                num_outputs,
            )
        )

    def register_conv2d_num_inputs_drr_classes(num_inputs):
        def register_conv2d_num_outputs_drr_classes(num_outputs):
            return register_conv2d_drr_class(num_inputs + 2, num_outputs + 1)

        ap.map(register_conv2d_num_outputs_drr_classes, range(max_num_outputs))

    ap.map(register_conv2d_num_inputs_drr_classes, range(max_num_inputs))


register_conv2d_epilogue_class(
    base_class=Conv2dEpilogueFusion, max_num_inputs=10, max_num_outputs=10
)
