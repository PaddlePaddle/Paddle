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
import conv2d_variadic_tpl
import epilogue_access_topo_simplify
import index_program_translator_util
import ir_tools
import kernel_arg_id_util
import kernel_arg_translator_util  # noqa: F401
import low_level_ir_code_gen_ctx_util  # noqa: F401
import op_compute_translator_util
import op_conversion_drr_pass  # noqa: F401
import pir  # noqa: F401
import program_translator_util
import topo_drr_pass
import umprime  # noqa: F401
import variadic_mixin


class Conv2dEpilogueFusion(abstract_drr.DrrPass):
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
        program = ir_tools.copy_fused_ops_to_program(
            o.trivial_op, tensor_match_ctx=t
        )
        program = epilogue_access_topo_simplify.simplify_epilogue_program(
            program,
            anchor_data_op_name="conv2d_out",
            number_of_inputs=self.number_of_inputs(),
            number_of_outputs=self.number_of_outputs(),
        )
        # The cutlass implicit gemm backend only supports the NHWC activation,
        # there is no NCHW conv2d fprop iterator in cutlass.
        return (
            program.empty()
            if o.conv2d_op.data_format.match(a_str=lambda x: x) == "NHWC"
            else False
        )

    def _insert_load_from_global(self, program, input_names):
        init_pass_manager = ir_tools.create_pass_manager()

        def AddPass(input_name):
            ir_pass = topo_drr_pass.InitNaiveLoadFromGlobalAccessTopoPass(
                input_name
            )
            init_pass_manager.add_pass(
                ir_tools.create_access_topo_drr_one_step_pass(ir_pass)
            )

        ap.map(AddPass, input_names)
        init_pass_manager.run(program)

    def _insert_store_to_global(self, program, output_names):
        init_pass_manager = ir_tools.create_pass_manager()
        ir_pass = topo_drr_pass.FakeDataStoreToGlobalForYieldAccessTopoPass(
            output_names
        )
        init_pass_manager.add_pass(
            ir_tools.create_access_topo_drr_one_step_pass(ir_pass)
        )
        init_pass_manager.run(program)

    def _make_kernel_arg_translator(self):
        return conv2d_variadic_tpl.make_kernel_arg_translator()

    def _get_int_attr(self, attr):
        return attr.match(a_i32=lambda x: int(x))

    def _get_int_list_attr(self, attr):
        # `int[]` of pd_op.conv2d is ArrayAttribute<Int32Attribute>.
        return attr.match(
            a_array=lambda values: ap.map(self._get_int_attr, values)
        )

    def _apply_topo_access_passes(self, mut_program, anchor_data_op_name):
        init_pass_manager = ir_tools.create_pass_manager()
        init_down_spider = topo_drr_pass.InitDownSpiderAccessTopoPass(
            anchor_data_op_name
        )
        init_pass_manager.add_pass(
            ir_tools.create_access_topo_drr_one_step_pass(init_down_spider)
        )
        init_pass_manager.run(mut_program)
        pass_manager = ir_tools.create_pass_manager()
        pass_manager.add_pass(ir_tools.create_access_topo_drr_pass("default"))
        pass_manager.add_pass(ir_tools.create_dce_pass())
        pass_manager.run(mut_program)

    def _simplify_index_program(self, mut_program):
        pass_manager = ir_tools.create_pass_manager()
        drr_pass = topo_drr_pass.ConvertUpSpiderStoreDataOpToYieldOpPass()
        pass_manager.add_pass(
            ir_tools.create_access_topo_drr_one_step_pass(drr_pass)
        )
        drr_pass = topo_drr_pass.ConvertDownSpiderStoreDataOpToYieldOpPass()
        pass_manager.add_pass(
            ir_tools.create_access_topo_drr_one_step_pass(drr_pass)
        )
        pass_manager.add_pass(ir_tools.create_dce_pass())
        pass_manager.run(mut_program)
        return mut_program

    def _make_index_func_unique_id2index_program(
        self, compute_program, anchor_data_op_name, input_names, output_names
    ):
        full_index_program = compute_program.clone()
        self._apply_topo_access_passes(full_index_program, anchor_data_op_name)
        print('full_index_program: ', full_index_program)

        def MatchAndCopyInputIndex(dst_input_name):
            pass_manager = ir_tools.create_pass_manager()
            removed_programs = ap.MutableList()
            rm_elementwise_drr_pass = (
                epilogue_access_topo_simplify.RemoveElementInputIndexPass(
                    src_data_op_name=anchor_data_op_name,
                    dst_load_from_global_op_name=dst_input_name,
                )
            )
            rm_elementwise_ir_pass = (
                ir_tools.create_access_topo_drr_one_step_pass(
                    rm_elementwise_drr_pass,
                    matched_pattern_mut_list=removed_programs,
                )
            )
            pass_manager.add_pass(rm_elementwise_ir_pass)
            rm_broadcast_drr_pass = (
                epilogue_access_topo_simplify.RemoveBroadcastInputIndexPass(
                    src_data_op_name=anchor_data_op_name,
                    dst_load_from_global_op_name=dst_input_name,
                )
            )
            rm_broadcast_ir_pass = (
                ir_tools.create_access_topo_drr_one_step_pass(
                    rm_broadcast_drr_pass,
                    matched_pattern_mut_list=removed_programs,
                )
            )
            pass_manager.add_pass(rm_broadcast_ir_pass)
            pass_manager.run(full_index_program)

            def Converter(program):
                return [dst_input_name, self._simplify_index_program(program)]

            return ap.map(Converter, removed_programs)

        input_and_index_programs = ap.flat_map(
            MatchAndCopyInputIndex, input_names
        )

        def MatchAndCopyOutputIndex(dst_output_name):
            print('full_index_program output: ', full_index_program)
            pass_manager = ir_tools.create_pass_manager()
            removed_programs = ap.MutableList()
            drr_pass = epilogue_access_topo_simplify.RemoveOutputIndexPass(
                src_data_op_name=anchor_data_op_name,
                dst_store_to_global_op_name=dst_output_name,
            )
            ir_pass = ir_tools.create_access_topo_drr_one_step_pass(
                drr_pass, matched_pattern_mut_list=removed_programs
            )
            pass_manager.add_pass(ir_pass)
            pass_manager.run(full_index_program)

            def Converter(program):
                return [dst_output_name, self._simplify_index_program(program)]

            print('len removed of output: ', len(removed_programs))
            return ap.map(Converter, removed_programs)

        output_and_index_programs = ap.flat_map(
            MatchAndCopyOutputIndex, output_names
        )
        return ap.OrderedDict(
            [*input_and_index_programs, *output_and_index_programs]
        )

    def _replace_with_load_from_register(
        self, mut_program, load_ir_value_name, register_var_name
    ):
        pass_manager = ir_tools.create_pass_manager()
        drr_pass = topo_drr_pass.ReplaceWithLoadFromRegisterPass(
            name=load_ir_value_name, register_var_name=register_var_name
        )
        pass_manager.add_pass(
            ir_tools.create_access_topo_drr_one_step_pass(drr_pass)
        )
        pass_manager.add_pass(ir_tools.create_dce_pass())
        pass_manager.run(mut_program)
        return mut_program

    def _replace_with_store_to_register(
        self, mut_program, store_ir_value_name, register_var_name
    ):
        pass_manager = ir_tools.create_pass_manager()
        drr_pass = topo_drr_pass.ReplaceWithStoreToRegisterPass(
            name=store_ir_value_name, register_var_name=register_var_name
        )
        pass_manager.add_pass(
            ir_tools.create_access_topo_drr_one_step_pass(drr_pass)
        )
        pass_manager.add_pass(ir_tools.create_dce_pass())
        pass_manager.run(mut_program)
        return mut_program

    def _get_program_translator(self, ctx, o, t):
        outputs_name_list = ap.map(
            lambda i: f"output{i}", range(self.number_of_outputs())
        )
        other_outputs_name_list = ap.map(
            lambda i: f"output{i + 1}", range(self.number_of_outputs() - 1)
        )
        local_outputs_name_list = ap.map(
            lambda i: f"out{i}", range(self.number_of_outputs())
        )
        inputs_name_list = (
            ap.map(
                lambda i: f"input{i + 2}", range(self.number_of_inputs() - 2)
            )
            if self.number_of_inputs() > 2
            else []
        )
        mut_program = ir_tools.copy_fused_ops_to_program(
            o.trivial_op, tensor_match_ctx=t
        )
        print("before-umprime: ", mut_program)
        pass_manager = ir_tools.create_pass_manager()
        pass_manager.add_pass(ir_tools.create_access_topo_drr_pass("umprime"))
        pass_manager.add_pass(ir_tools.create_dce_pass())
        pass_manager.run(mut_program)
        self._insert_load_from_global(mut_program, input_names=["conv2d_out"])
        self._insert_load_from_global(mut_program, input_names=inputs_name_list)
        self._insert_store_to_global(
            mut_program, output_names=outputs_name_list
        )
        kernel_arg_translator = self._make_kernel_arg_translator()
        index_func_unique_id2index_program = (
            self._make_index_func_unique_id2index_program(
                mut_program,
                anchor_data_op_name="conv2d_out",
                input_names=inputs_name_list,
                output_names=other_outputs_name_list,
            )
        )
        print(
            "index_func_unique_id2index_program:\n",
            index_func_unique_id2index_program,
        )
        index_program_translator_map = index_program_translator_util.IndexProgramTranslatorMap(
            index_func_unique_id2index_program=index_func_unique_id2index_program,
            kernel_arg_translator=kernel_arg_translator,
            anchor_iter_var_names=conv2d_variadic_tpl.get_anchor_iter_var_names(),
        )
        self._replace_with_load_from_register(
            mut_program,
            load_ir_value_name="conv2d_out",
            register_var_name="x",
        )
        self._replace_with_store_to_register(mut_program, "output0", "out")
        print("mut_program:", mut_program)
        op_compute_translator_maker = (
            op_compute_translator_util.OpComputeTranslatorFactory()
        )
        program_translator = program_translator_util.ProgramTranslator(
            program_property=mut_program.copy_to_const_program_data(),
            kernel_arg_translator=kernel_arg_translator,
            index_program_translator_map=index_program_translator_map,
            op_translator_maker=op_compute_translator_maker,
        )

        return program_translator

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
