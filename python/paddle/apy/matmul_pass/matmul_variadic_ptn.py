# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import epilogue_access_topo_simplify
import ir_tools
import kernel_arg_id_util
import kernel_arg_translator_util  # noqa: F401
import low_level_ir_code_gen_ctx_util  # noqa: F401
import matmul_variadic_tpl
import op_conversion_drr_pass  # noqa: F401
import pir  # noqa: F401
import umprime  # noqa: F401
import variadic_mixin


class MatmulEpilogueFusion(variadic_mixin.VariadicEpilogueFusionBase):
    def source_pattern(self, o, t):
        in_num = self.number_of_inputs()
        out_num = self.number_of_outputs()
        o.matmul_op = o.ap_native_op("pd_op.matmul")
        o.matmul_op([t.input0, t.input1], [t.mm_out])
        o.trivial_op = o.ap_trivial_fusion_op()
        o.trivial_op(
            [
                t.mm_out,
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
            self._epilogue_can_fuse(o, t)
        )

    def _is_supported_by_device(self):
        compile_command_generator = (
            compile_command_util.CompileCommandGenerator()
        )
        return compile_command_generator.supports("matmul")

    def _epilogue_can_fuse(self, o, t):
        program = ir_tools.copy_fused_ops_to_program(
            o.trivial_op, tensor_match_ctx=t
        )
        program = epilogue_access_topo_simplify.simplify_epilogue_program(
            program,
            anchor_data_op_name="mm_out",
            number_of_inputs=self.number_of_inputs(),
            number_of_outputs=self.number_of_outputs(),
        )
        return program.empty()

    def _anchor_data_op_name(self):
        return "mm_out"

    def _anchor_input_count(self):
        return 2

    def _anchor_iter_var_names(self):
        return matmul_variadic_tpl.get_anchor_iter_var_names()

    def _make_kernel_arg_translator(self):
        return matmul_variadic_tpl.make_kernel_arg_translator()

    def code_gen(self, ctx, o, t):
        program_translator = self._get_program_translator(ctx, o, t)
        mut_kernel_arg_id_registry = kernel_arg_id_util.KernelArgIdNameRegistry(
            code_gen_ctx=ctx, tensor_match_ctx=t, name_prefix=""
        )

        template_module = matmul_variadic_tpl.MatmulVariadicTemplate(
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
        )


def register_matmul_epilogue_class(base_class, max_num_inputs, max_num_outputs):
    def register_drr_class(num_inputs, num_outputs):
        abstract_drr.register_drr_pass(
            f"matmul_epilogue_in{num_inputs}_out{num_outputs}_fusion", nice=0
        )(
            variadic_mixin.get_mixin_class(
                base_class,
                "MatmulEpilogueFusion",
                num_inputs,
                num_outputs,
            )
        )

    def register_num_inputs_drr_classes(num_inputs):
        def register_num_outputs_drr_classes(num_outputs):
            return register_drr_class(num_inputs + 2, num_outputs + 1)

        ap.map(register_num_outputs_drr_classes, range(max_num_outputs))

    ap.map(register_num_inputs_drr_classes, range(max_num_inputs))


register_matmul_epilogue_class(
    base_class=MatmulEpilogueFusion, max_num_inputs=10, max_num_outputs=10
)
