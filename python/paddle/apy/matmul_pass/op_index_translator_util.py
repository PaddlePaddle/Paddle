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
import index_code_gen_value_util


class PdOpDataCodeGen:
    def __init__(
        self,
        index_program_id,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        anchor_iter_var_names,
    ):
        self.index_program_id = index_program_id
        self.op_property = op_property
        self.input_properties = input_properties
        self.output_properties = output_properties
        self.kernel_arg_translator = kernel_arg_translator
        self.anchor_iter_var_names = anchor_iter_var_names

    def __call__(
        self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
    ):
        return [
            index_code_gen_value_util.IndexCodeGenValue(
                self.anchor_iter_var_names
            )
        ]


class PdOpFullIntArrayCodeGen:
    def __init__(
        self,
        index_program_id,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        anchor_iter_var_names,
    ):
        self.index_program_id = index_program_id
        self.op_property = op_property
        self.input_properties = input_properties
        self.output_properties = output_properties
        self.kernel_arg_translator = kernel_arg_translator
        self.anchor_iter_var_names = anchor_iter_var_names

    def __call__(
        self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
    ):
        out = index_code_gen_value_util.IndexCodeGenValue(None)

        def get_int64(attr):
            return attr.match(a_i64=lambda x: x)

        def convert_list(lst):
            return ap.map(get_int64, lst)

        out.const_data = self.op_property.attributes.value.match(
            a_array=convert_list
        )
        return [out]


class PdOpSumCodeGen:
    def __init__(
        self,
        index_program_id,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        anchor_iter_var_names,
    ):
        self.index_program_id = index_program_id
        self.op_property = op_property
        self.input_properties = input_properties
        self.output_properties = output_properties
        self.kernel_arg_translator = kernel_arg_translator
        self.anchor_iter_var_names = anchor_iter_var_names

    def __call__(
        self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
    ):
        input_iter_var_names = inputs[0].iter_var_names
        reduced_axes_set = ap.OrderedDict(
            ap.map(lambda x: [int(x), True], inputs[1].const_data)
        )
        non_reduced_axes = ap.filter(
            lambda x: reduced_axes_set.contains(x) == False,  # noqa: E712
            range(len(input_iter_var_names)),
        )
        output_iter_var_names = ap.map(
            lambda i: input_iter_var_names[i], non_reduced_axes
        )
        return [
            index_code_gen_value_util.IndexCodeGenValue(output_iter_var_names)
        ]


class CinnOpReshapeCodeGen:
    def __init__(
        self,
        index_program_id,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        anchor_iter_var_names,
    ):
        self.index_program_id = index_program_id
        self.op_property = op_property
        self.input_properties = input_properties
        self.output_properties = output_properties
        self.kernel_arg_translator = kernel_arg_translator
        self.anchor_iter_var_names = anchor_iter_var_names

    def __call__(
        self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
    ):
        symbolic_shape = self.input_properties[0].symbolic_shape

        def get_or_create_dim_var_name(dim_expr):
            arg_var_name = mut_kernel_arg_id_registry.get_dim_expr_var_name(
                dim_expr
            )
            return self.kernel_arg_translator.get_use_name(arg_var_name)

        def get_dim_var_name(i):
            dim_expr = symbolic_shape[i]
            return get_or_create_dim_var_name(dim_expr)

        rank = len(symbolic_shape)
        stride_dims_list = ap.map(
            lambda num_dims: ap.map(
                lambda i: get_dim_var_name(num_dims + i + 1),
                range(rank - 1 - num_dims),
            ),
            range(rank),
        )
        var_name_and_dims_list = ap.map(
            lambda pair: [pair[0], *pair[1]],
            zip(inputs[0].iter_var_names, stride_dims_list),
        )
        offset_expr = " + ".join(
            ap.map(lambda elts: " * ".join(elts), var_name_and_dims_list)
        )
        assert (
            len(self.output_properties[0].symbolic_shape) == 1
        ), "len(self.output_properties[0]) should be 1"
        return [
            index_code_gen_value_util.IndexCodeGenValue([f"({offset_expr})"])
        ]


class CfYieldCodeGen:
    def __init__(
        self,
        index_program_id,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        anchor_iter_var_names,
    ):
        self.index_program_id = index_program_id
        self.op_property = op_property
        self.input_properties = input_properties
        self.output_properties = output_properties
        self.kernel_arg_translator = kernel_arg_translator
        self.anchor_iter_var_names = anchor_iter_var_names

    def __call__(
        self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
    ):
        return []


class OpIndexTranslatorFactory:
    def __init__(self):
        self.op_name2class = ap.OrderedDict(
            [
                ["pd_op.data", PdOpDataCodeGen],
                ["pd_op.full_int_array", PdOpFullIntArrayCodeGen],
                ["pd_op.sum", PdOpSumCodeGen],
                ["cinn_op.reshape", CinnOpReshapeCodeGen],
                ["cf.yield", CfYieldCodeGen],
            ]
        )

    def __call__(
        self,
        index_program_id,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        anchor_iter_var_names,
    ):
        cls = self._get_class(op_property.op_name)
        return cls(
            index_program_id=index_program_id,
            op_property=op_property,
            input_properties=input_properties,
            output_properties=output_properties,
            kernel_arg_translator=kernel_arg_translator,
            anchor_iter_var_names=anchor_iter_var_names,
        )

    def _get_class(self, op_name):
        return self.op_name2class[op_name]
