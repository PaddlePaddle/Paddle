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
import epilogue_access_topo_simplify
import index_program_translator_util
import ir_tools
import kernel_arg_translator_util  # noqa: F401
import low_level_ir_code_gen_ctx_util  # noqa: F401
import op_compute_translator_util
import op_conversion_drr_pass  # noqa: F401
import pir  # noqa: F401
import program_translator_util
import topo_drr_pass
import umprime  # noqa: F401


class NumberOfInputsTrait0:
    def number_of_inputs(self):
        return 0


class NumberOfInputsTrait1:
    def number_of_inputs(self):
        return 1


class NumberOfInputsTrait2:
    def number_of_inputs(self):
        return 2


class NumberOfInputsTrait3:
    def number_of_inputs(self):
        return 3


class NumberOfInputsTrait4:
    def number_of_inputs(self):
        return 4


class NumberOfInputsTrait5:
    def number_of_inputs(self):
        return 5


class NumberOfInputsTrait6:
    def number_of_inputs(self):
        return 6


class NumberOfInputsTrait7:
    def number_of_inputs(self):
        return 7


class NumberOfInputsTrait8:
    def number_of_inputs(self):
        return 8


class NumberOfInputsTrait9:
    def number_of_inputs(self):
        return 9


class NumberOfInputsTrait10:
    def number_of_inputs(self):
        return 10


class NumberOfInputsTrait11:
    def number_of_inputs(self):
        return 11


class NumberOfInputsTrait12:
    def number_of_inputs(self):
        return 12


class NumberOfInputsTrait13:
    def number_of_inputs(self):
        return 13


class NumberOfInputsTrait14:
    def number_of_inputs(self):
        return 14


class NumberOfInputsTrait15:
    def number_of_inputs(self):
        return 15


class NumberOfInputsTrait16:
    def number_of_inputs(self):
        return 16


class NumberOfInputsTrait17:
    def number_of_inputs(self):
        return 17


class NumberOfOutputsTrait0:
    def number_of_outputs(self):
        return 0


class NumberOfOutputsTrait1:
    def number_of_outputs(self):
        return 1


class NumberOfOutputsTrait2:
    def number_of_outputs(self):
        return 2


class NumberOfOutputsTrait3:
    def number_of_outputs(self):
        return 3


class NumberOfOutputsTrait4:
    def number_of_outputs(self):
        return 4


class NumberOfOutputsTrait5:
    def number_of_outputs(self):
        return 5


class NumberOfOutputsTrait6:
    def number_of_outputs(self):
        return 6


class NumberOfOutputsTrait7:
    def number_of_outputs(self):
        return 7


class NumberOfOutputsTrait8:
    def number_of_outputs(self):
        return 8


class NumberOfOutputsTrait9:
    def number_of_outputs(self):
        return 9


class NumberOfOutputsTrait10:
    def number_of_outputs(self):
        return 10


class NumberOfOutputsTrait11:
    def number_of_outputs(self):
        return 11


class NumberOfOutputsTrait12:
    def number_of_outputs(self):
        return 12


class NumberOfOutputsTrait13:
    def number_of_outputs(self):
        return 13


class NumberOfOutputsTrait14:
    def number_of_outputs(self):
        return 14


class NumberOfOutputsTrait15:
    def number_of_outputs(self):
        return 15


class NumberOfOutputsTrait16:
    def number_of_outputs(self):
        return 16


class NumberOfOutputsTrait17:
    def number_of_outputs(self):
        return 17


class NumberOfOutputsTrait18:
    def number_of_outputs(self):
        return 18


class NumberOfOutputsTrait19:
    def number_of_outputs(self):
        return 19


class NumberOfOutputsTrait20:
    def number_of_outputs(self):
        return 20


class NumberOfOutputsTrait21:
    def number_of_outputs(self):
        return 21


class NumberOfOutputsTrait22:
    def number_of_outputs(self):
        return 22


class VariadicEpilogueFusionBase(abstract_drr.DrrPass):
    # Shared pipeline for `<anchor op> + trivial fusion` epilogue fusion:
    # copy the fused elementwise program to a standalone program, turn the
    # anchor output and the extra inputs into global loads, then translate
    # the index/compute programs into the backend template.
    #
    # Subclasses only declare the anchor-specific bits:
    #   - _anchor_data_op_name: anchor data op name, e.g. "mm_out".
    #   - _anchor_input_count: number of anchor operands consumed by the
    #     fusion, e.g. 2 for matmul / conv2d and 3 for attention Q/K/V.
    #   - _anchor_iter_var_names: anchor iteration variable names used by
    #     `IndexProgramTranslatorMap`.
    #   - _make_kernel_arg_translator: the backend template module's
    #     kernel-arg translator.
    # `source_pattern` / `result_pattern` / `constraint` / `code_gen` stay in
    # the subclasses because they carry the op-specific matching and codegen.
    def _anchor_data_op_name(self):
        return ""

    def _anchor_input_count(self):
        return 0

    def _anchor_iter_var_names(self):
        return []

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
        anchor_data_op_name = self._anchor_data_op_name()
        anchor_input_count = self._anchor_input_count()
        outputs_name_list = ap.map(
            lambda i: f"output{i}", range(self.number_of_outputs())
        )
        other_outputs_name_list = ap.map(
            lambda i: f"output{i + 1}", range(self.number_of_outputs() - 1)
        )
        inputs_name_list = (
            ap.map(
                lambda i: f"input{i + anchor_input_count}",
                range(self.number_of_inputs() - anchor_input_count),
            )
            if self.number_of_inputs() > anchor_input_count
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
        self._insert_load_from_global(
            mut_program, input_names=[anchor_data_op_name]
        )
        self._insert_load_from_global(mut_program, input_names=inputs_name_list)
        self._insert_store_to_global(
            mut_program, output_names=outputs_name_list
        )
        kernel_arg_translator = self._make_kernel_arg_translator()
        index_func_unique_id2index_program = (
            self._make_index_func_unique_id2index_program(
                mut_program,
                anchor_data_op_name=anchor_data_op_name,
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
            anchor_iter_var_names=self._anchor_iter_var_names(),
        )
        self._replace_with_load_from_register(
            mut_program,
            load_ir_value_name=anchor_data_op_name,
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


def get_mixin_class(
    base_class, class_name_prefix, number_of_inputs, number_of_outputs
):
    num_inputs_to_input_trait_class = [
        None,
        NumberOfInputsTrait1,
        NumberOfInputsTrait2,
        NumberOfInputsTrait3,
        NumberOfInputsTrait4,
        NumberOfInputsTrait5,
        NumberOfInputsTrait6,
        NumberOfInputsTrait7,
        NumberOfInputsTrait8,
        NumberOfInputsTrait9,
        NumberOfInputsTrait10,
        NumberOfInputsTrait11,
        NumberOfInputsTrait12,
        NumberOfInputsTrait13,
        NumberOfInputsTrait14,
        NumberOfInputsTrait15,
        NumberOfInputsTrait16,
        NumberOfInputsTrait17,
    ]
    num_outputs_to_output_trait_class = [
        None,
        NumberOfOutputsTrait1,
        NumberOfOutputsTrait2,
        NumberOfOutputsTrait3,
        NumberOfOutputsTrait4,
        NumberOfOutputsTrait5,
        NumberOfOutputsTrait6,
        NumberOfOutputsTrait7,
        NumberOfOutputsTrait8,
        NumberOfOutputsTrait9,
        NumberOfOutputsTrait10,
        NumberOfOutputsTrait11,
        NumberOfOutputsTrait12,
        NumberOfOutputsTrait13,
        NumberOfOutputsTrait14,
        NumberOfOutputsTrait15,
        NumberOfOutputsTrait16,
        NumberOfOutputsTrait17,
        NumberOfOutputsTrait18,
        NumberOfOutputsTrait19,
        NumberOfOutputsTrait20,
        NumberOfOutputsTrait21,
        NumberOfOutputsTrait22,
    ]
    return type(
        f"{class_name_prefix}{number_of_inputs}_{number_of_outputs}",
        [
            base_class,
            num_inputs_to_input_trait_class[number_of_inputs],
            num_outputs_to_output_trait_class[number_of_outputs],
        ],
        ap.SerializableAttrMap(),
    )
