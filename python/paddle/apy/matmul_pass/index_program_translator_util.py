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

import ap
import index_drr_pass_util
import ir_tools
import op_index_translator_util


class IndexProgramTranslatorMap:

    def __init__(
        self,
        index_func_unique_id2index_program,
        kernel_arg_translator,
        anchor_iter_var_names,
    ):
        self.kernel_arg_translator = kernel_arg_translator
        self.anchor_iter_var_names = anchor_iter_var_names
        items = index_func_unique_id2index_program.items()
        self.index_func_unique_id2translator = ap.OrderedDict(
            ap.map(
                lambda i: [items[i][0], self.make_translator(i, items[i][1])],
                range(len(items)),
            )
        )

    def get_offset_var_name(
        self,
        index_func_unique_id,
        mut_kernel_arg_id_registry,
        mut_lir_code_gen_ctx,
    ):
        translator = self.index_func_unique_id2translator[index_func_unique_id]
        ret = translator.translate(
            mut_kernel_arg_id_registry=mut_kernel_arg_id_registry,
            mut_lir_code_gen_ctx=mut_lir_code_gen_ctx,
        )
        return ret.iter_var_names[0]

    def make_translator(self, program_id, index_program):
        pass_manager = ir_tools.create_pass_manager()
        drr_pass = index_drr_pass_util.InsertReshapeBeforeYieldPass()
        pass_manager.add_pass(
            ir_tools.create_access_topo_drr_one_step_pass(drr_pass)
        )
        pass_manager.add_pass(ir_tools.create_dce_pass())
        pass_manager.run(index_program)
        return IndexProgramTranslator(
            index_program,
            program_id=program_id,
            kernel_arg_translator=self.kernel_arg_translator,
            index_op_translator_maker=op_index_translator_util.OpIndexTranslatorFactory(),
            anchor_iter_var_names=self.anchor_iter_var_names,
        )


class IndexProgramTranslator:

    def __init__(
        self,
        index_program,
        program_id,
        kernel_arg_translator,
        index_op_translator_maker,
        anchor_iter_var_names,
    ):
        self.program_id = program_id
        self.program_property = index_program.copy_to_const_program_data()
        self.kernel_arg_translator = kernel_arg_translator
        self.index_op_translator_maker = index_op_translator_maker
        self.anchor_iter_var_names = anchor_iter_var_names
        self.ir_value_index2translated_value = ap.MutableList()

        def PushNone(x):
            self.ir_value_index2translated_value.append(None)

        ap.map(PushNone, self.program_property.values)

    def translate(
        self,
        mut_kernel_arg_id_registry,
        mut_lir_code_gen_ctx,
    ):
        def TranslateOp(op_property):
            self._translate_op(
                op_property, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
            )

        ap.map(TranslateOp, self.program_property.ops)
        return self.ir_value_index2translated_value[-1]

    def _translate_op(
        self, op_property, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
    ):
        index_op_translator = self.index_op_translator_maker(
            index_program_id=self.program_id,
            op_property=op_property,
            input_properties=ap.map(
                self._get_value_property, op_property.input_value_indexes
            ),
            output_properties=ap.map(
                self._get_value_property, op_property.output_value_indexes
            ),
            kernel_arg_translator=self.kernel_arg_translator,
            anchor_iter_var_names=self.anchor_iter_var_names,
        )
        inputs = ap.map(
            self._get_translated_value, op_property.input_value_indexes
        )
        outputs = index_op_translator(
            inputs,
            mut_kernel_arg_id_registry=mut_kernel_arg_id_registry,
            mut_lir_code_gen_ctx=mut_lir_code_gen_ctx,
        )
        ap.map(
            self._set_translated_value,
            zip(op_property.output_value_indexes, outputs),
        )

    def _get_value_property(self, i):
        return self.program_property.values[i]

    def _get_translated_value(self, i):
        return self.ir_value_index2translated_value[i]

    def _set_translated_value(self, pair):
        self.ir_value_index2translated_value[pair[0]] = pair[1]
