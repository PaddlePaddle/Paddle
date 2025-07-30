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
import code_gen_value_util


class ApOpLoadFromRegisterCodeGen:
    def __init__(
        self,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        index_program_translator_map,
    ):
        self.op_property = op_property
        self.input_properties = input_properties
        self.output_properties = output_properties
        self.kernel_arg_translator = kernel_arg_translator
        self.index_program_translator_map = index_program_translator_map

    def __call__(
        self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
    ):
        out = self.get_out_cg_val(0)
        return [out]

    def get_out_cg_val(self, i):
        register_var_name_attr = self.op_property.attributes.register_var_name
        register_var_name = register_var_name_attr.match(a_str=lambda x: x)
        return code_gen_value_util.CodeGenValue(
            self.output_properties[i].type, register_var_name
        )


class ApOpLoadFromGlobalCodeGen:
    def __init__(
        self,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        index_program_translator_map,
    ):
        self.op_property = op_property
        self.input_properties = input_properties
        self.output_properties = output_properties
        self.kernel_arg_translator = kernel_arg_translator
        self.index_program_translator_map = index_program_translator_map

    def __call__(
        self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
    ):
        index_func_unique_id_attr = (
            self.op_property.attributes.index_func_unique_id
        )
        index_func_unique_id = index_func_unique_id_attr.match(
            a_str=lambda x: x
        )
        offset_var_name = self.index_program_translator_map.get_offset_var_name(
            index_func_unique_id=index_func_unique_id,
            mut_kernel_arg_id_registry=mut_kernel_arg_id_registry,
            mut_lir_code_gen_ctx=mut_lir_code_gen_ctx,
        )
        data_op_name = inputs[0].var_name
        arg_name = mut_kernel_arg_id_registry.get_in_tensor_data_ptr_var_name(
            data_op_name
        )
        ptr_var_name = self.kernel_arg_translator.get_use_name(arg_name)
        out = self.get_out_cg_val(0)
        mut_lir_code_gen_ctx.let(out, f"{ptr_var_name}[{offset_var_name}]")
        return [out]

    def get_out_cg_val(self, i):
        return code_gen_value_util.CodeGenValue(
            self.output_properties[i].type,
            f"op{self.op_property.op_index}_out{i}",
        )


class ApOpStoreToRegisterCodeGen:
    def __init__(
        self,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        index_program_translator_map,
    ):
        self.op_property = op_property
        self.input_properties = input_properties
        self.output_properties = output_properties
        self.kernel_arg_translator = kernel_arg_translator
        self.index_program_translator_map = index_program_translator_map

    def __call__(
        self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
    ):
        mut_lir_code_gen_ctx.stmts.append(
            f"{self.get_out_var_name()} = {inputs[0].var_name};"
        )
        return []

    def get_out_var_name(self):
        register_var_name_attr = self.op_property.attributes.register_var_name
        return register_var_name_attr.match(a_str=lambda x: x)


class ApOpStoreToGlobalCodeGen:
    def __init__(
        self,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        index_program_translator_map,
    ):
        self.op_property = op_property
        self.input_properties = input_properties
        self.output_properties = output_properties
        self.kernel_arg_translator = kernel_arg_translator
        self.index_program_translator_map = index_program_translator_map
        self.ptr_type2data_type = ap.OrderedDict(
            [
                [ap.PointerType.const_float_ptr, "const float"],
                [ap.PointerType.const_float16_ptr, "const half"],
                [ap.PointerType.float_ptr, "float"],
                [ap.PointerType.float16_ptr, "half"],
            ]
        )

    def __call__(
        self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
    ):
        index_func_unique_id_attr = (
            self.op_property.attributes.index_func_unique_id
        )
        index_func_unique_id = index_func_unique_id_attr.match(
            a_str=lambda x: x
        )
        offset_var_name = self.index_program_translator_map.get_offset_var_name(
            index_func_unique_id=index_func_unique_id,
            mut_kernel_arg_id_registry=mut_kernel_arg_id_registry,
            mut_lir_code_gen_ctx=mut_lir_code_gen_ctx,
        )
        arg_name = mut_kernel_arg_id_registry.get_out_tensor_data_ptr_var_name(
            index_func_unique_id
        )
        glb_data_type = self.get_glb_type(
            mut_kernel_arg_id_registry, index_func_unique_id
        )
        ptr_var_name = self.kernel_arg_translator.get_use_name(arg_name)
        mut_lir_code_gen_ctx.store(
            glb_data_type, ptr_var_name, offset_var_name, inputs[1].var_name
        )
        return []

    def get_glb_type(self, mut_kernel_arg_id_registry, index_func_unique_id):
        arg_name = mut_kernel_arg_id_registry.get_out_tensor_data_ptr_var_name(
            index_func_unique_id
        )
        kernel_arg_name2ids = ap.OrderedDict(
            ap.map(
                lambda item: [item[1], item[0]],
                mut_kernel_arg_id_registry.generated_kernel_arg_id2unique_name.items(),
            )
        )
        kernel_arg_id = kernel_arg_name2ids[arg_name]
        return self.ptr_type2data_type[kernel_arg_id.type]

    def get_out_var_name(self):
        register_var_name_attr = self.op_property.attributes.register_var_name
        return register_var_name_attr.match(a_str=lambda x: x)


class PdOpDataCodeGen:
    def __init__(
        self,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        index_program_translator_map,
    ):
        self.op_property = op_property
        self.input_properties = input_properties
        self.output_properties = output_properties
        self.kernel_arg_translator = kernel_arg_translator
        self.index_program_translator_map = index_program_translator_map

    def __call__(
        self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
    ):
        out = self.get_out_cg_val(0)
        return [out]

    def get_out_cg_val(self, i):
        name = self.op_property.attributes.name.match(a_str=lambda x: x)
        return code_gen_value_util.CodeGenValue(
            self.output_properties[i].type, name
        )


class PdOpFullCodeGen:
    def __init__(
        self,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        index_program_translator_map,
    ):
        self.op_property = op_property
        self.input_properties = input_properties
        self.output_properties = output_properties
        self.kernel_arg_translator = kernel_arg_translator
        self.index_program_translator_map = index_program_translator_map

    def __call__(
        self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
    ):
        value = self.op_property.attributes.value.match(a_f64=lambda x: x)
        out = self.get_out_cg_val(0)
        mut_lir_code_gen_ctx.let(out, f"{value}")
        return [out]

    def get_out_cg_val(self, i):
        return code_gen_value_util.CodeGenValue(
            self.output_properties[i].type,
            f"op{self.op_property.op_index}_out{i}",
        )


class PdOpCastCodeGen:
    def __init__(
        self,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        index_program_translator_map,
    ):
        self.op_property = op_property
        self.input_properties = input_properties
        self.output_properties = output_properties
        self.kernel_arg_translator = kernel_arg_translator
        self.index_program_translator_map = index_program_translator_map
        self.dtype2type_name = ap.OrderedDict(
            [
                [ap.DataType.float, "float"],
                [ap.DataType.float16, "half"],
                [ap.DataType.bfloat16, "nv_bfloat16"],
                [ap.DataType.int32, "int"],
                [ap.DataType.int64, "int64_t"],
            ]
        )

    def __call__(
        self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
    ):
        dtype = self.op_property.attributes.dtype.match(a_dtype=lambda x: x)
        dtype_name = self.dtype2type_name[dtype]
        out = self.get_out_cg_val(0)
        mut_lir_code_gen_ctx.let(
            out, f"static_cast<{dtype_name}>({inputs[0].var_name})"
        )
        return [out]

    def get_out_cg_val(self, i):
        return code_gen_value_util.CodeGenValue(
            self.output_properties[i].type,
            f"op{self.op_property.op_index}_out{i}",
        )


class PdOpExpCodeGen:
    def __init__(
        self,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        index_program_translator_map,
    ):
        self.op_property = op_property
        self.input_properties = input_properties
        self.output_properties = output_properties
        self.kernel_arg_translator = kernel_arg_translator
        self.index_program_translator_map = index_program_translator_map

    def __call__(
        self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
    ):
        out = self.get_out_cg_val(0)
        mut_lir_code_gen_ctx.let(out, f"expf({inputs[0].var_name})")
        return [out]

    def get_out_cg_val(self, i):
        return code_gen_value_util.CodeGenValue(
            self.output_properties[i].type,
            f"op{self.op_property.op_index}_out{i}",
        )


class PdOpReluCodeGen:
    def __init__(
        self,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        index_program_translator_map,
    ):
        self.op_property = op_property
        self.input_properties = input_properties
        self.output_properties = output_properties
        self.kernel_arg_translator = kernel_arg_translator
        self.index_program_translator_map = index_program_translator_map

    def __call__(
        self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
    ):
        out = self.get_out_cg_val(0)
        mut_lir_code_gen_ctx.let(
            out, f"({inputs[0].var_name} > 0 ? {inputs[0].var_name} : 0) "
        )
        return [out]

    def get_out_cg_val(self, i):
        return code_gen_value_util.CodeGenValue(
            self.output_properties[i].type,
            f"op{self.op_property.op_index}_out{i}",
        )


class PdOpErfCodeGen:
    def __init__(
        self,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        index_program_translator_map,
    ):
        self.op_property = op_property
        self.input_properties = input_properties
        self.output_properties = output_properties
        self.kernel_arg_translator = kernel_arg_translator
        self.index_program_translator_map = index_program_translator_map

    def __call__(
        self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
    ):
        var_name = inputs[0].var_name
        out = self.get_out_cg_val(0)
        mut_lir_code_gen_ctx.let(out, f"erf({var_name})")
        return [out]

    def get_out_cg_val(self, i):
        return code_gen_value_util.CodeGenValue(
            self.output_properties[i].type,
            f"op{self.op_property.op_index}_out{i}",
        )


class PdOpElementwisePowCodeGen:
    def __init__(
        self,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        index_program_translator_map,
    ):
        self.op_property = op_property
        self.input_properties = input_properties
        self.output_properties = output_properties
        self.kernel_arg_translator = kernel_arg_translator
        self.index_program_translator_map = index_program_translator_map

    def __call__(
        self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
    ):
        exponent = inputs[1].var_name
        var_name = inputs[0].var_name
        out = self.get_out_cg_val(0)
        mut_lir_code_gen_ctx.let(out, f"ComputePow({var_name}, {exponent})")
        return [out]

    def get_out_cg_val(self, i):
        return code_gen_value_util.CodeGenValue(
            self.output_properties[i].type,
            f"op{self.op_property.op_index}_out{i}",
        )


class PdOpSinCodeGen:
    def __init__(
        self,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        index_program_translator_map,
    ):
        self.op_property = op_property
        self.input_properties = input_properties
        self.output_properties = output_properties
        self.kernel_arg_translator = kernel_arg_translator
        self.index_program_translator_map = index_program_translator_map

    def __call__(
        self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
    ):
        var_name = inputs[0].var_name
        out = self.get_out_cg_val(0)
        mut_lir_code_gen_ctx.let(out, f"sin({var_name})")
        return [out]

    def get_out_cg_val(self, i):
        return code_gen_value_util.CodeGenValue(
            self.output_properties[i].type,
            f"op{self.op_property.op_index}_out{i}",
        )


class PdOpTanhCodeGen:
    def __init__(
        self,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        index_program_translator_map,
    ):
        self.op_property = op_property
        self.input_properties = input_properties
        self.output_properties = output_properties
        self.kernel_arg_translator = kernel_arg_translator
        self.index_program_translator_map = index_program_translator_map

    def __call__(
        self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
    ):
        var_name = inputs[0].var_name
        out = self.get_out_cg_val(0)
        mut_lir_code_gen_ctx.let(out, f"tanh({var_name})")
        return [out]

    def get_out_cg_val(self, i):
        return code_gen_value_util.CodeGenValue(
            self.output_properties[i].type,
            f"op{self.op_property.op_index}_out{i}",
        )


class PdOpFloorCodeGen:
    def __init__(
        self,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        index_program_translator_map,
    ):
        self.op_property = op_property
        self.input_properties = input_properties
        self.output_properties = output_properties
        self.kernel_arg_translator = kernel_arg_translator
        self.index_program_translator_map = index_program_translator_map

    def __call__(
        self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
    ):
        var_name = inputs[0].var_name
        out = self.get_out_cg_val(0)
        mut_lir_code_gen_ctx.let(out, f"floor({var_name})")
        return [out]

    def get_out_cg_val(self, i):
        return code_gen_value_util.CodeGenValue(
            self.output_properties[i].type,
            f"op{self.op_property.op_index}_out{i}",
        )


class CinnOpScaleCodeGen:
    def __init__(
        self,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        index_program_translator_map,
    ):
        self.op_property = op_property
        self.input_properties = input_properties
        self.output_properties = output_properties
        self.kernel_arg_translator = kernel_arg_translator
        self.index_program_translator_map = index_program_translator_map

    def __call__(
        self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
    ):
        scale = self.op_property.attributes.scale.match(a_f32=lambda x: x)
        bias = self.op_property.attributes.bias.match(a_f32=lambda x: x)
        bias_after_scale = self.op_property.attributes.bias_after_scale.match(
            a_bool=lambda x: x
        )
        in_name = inputs[0].var_name
        true_str = f"{scale} * {in_name} + {bias}"
        false_str = f"{scale} * ({in_name} + {bias})"
        out = self.get_out_cg_val(0)
        mut_lir_code_gen_ctx.let(
            out, true_str if bias_after_scale else false_str
        )
        return [out]

    def get_out_cg_val(self, i):
        return code_gen_value_util.CodeGenValue(
            self.output_properties[i].type,
            f"op{self.op_property.op_index}_out{i}",
        )


class PdOpSubtractCodeGen:
    def __init__(
        self,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        index_program_translator_map,
    ):
        self.op_property = op_property
        self.input_properties = input_properties
        self.output_properties = output_properties
        self.kernel_arg_translator = kernel_arg_translator
        self.index_program_translator_map = index_program_translator_map

    def __call__(
        self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
    ):
        a = inputs[0]
        b = inputs[1]
        out = self.get_out_cg_val(0)
        mut_lir_code_gen_ctx.let(out, f"({a.var_name} - {b.var_name})")
        return [out]

    def get_out_cg_val(self, i):
        return code_gen_value_util.CodeGenValue(
            self.output_properties[i].type,
            f"op{self.op_property.op_index}_out{i}",
        )


class PdOpAddCodeGen:
    def __init__(
        self,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        index_program_translator_map,
    ):
        self.op_property = op_property
        self.input_properties = input_properties
        self.output_properties = output_properties
        self.kernel_arg_translator = kernel_arg_translator
        self.index_program_translator_map = index_program_translator_map

    def __call__(
        self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
    ):
        a = inputs[0]
        b = inputs[1]
        out = self.get_out_cg_val(0)
        mut_lir_code_gen_ctx.let(out, f"{a.var_name} + {b.var_name}")
        return [out]

    def get_out_cg_val(self, i):
        return code_gen_value_util.CodeGenValue(
            self.output_properties[i].type,
            f"op{self.op_property.op_index}_out{i}",
        )


class PdOpMultiplyCodeGen:
    def __init__(
        self,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        index_program_translator_map,
    ):
        self.op_property = op_property
        self.input_properties = input_properties
        self.output_properties = output_properties
        self.kernel_arg_translator = kernel_arg_translator
        self.index_program_translator_map = index_program_translator_map

    def __call__(
        self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
    ):
        a = inputs[0]
        b = inputs[1]
        out = self.get_out_cg_val(0)
        mut_lir_code_gen_ctx.let(out, f"{a.var_name} * {b.var_name}")
        return [out]

    def get_out_cg_val(self, i):
        return code_gen_value_util.CodeGenValue(
            self.output_properties[i].type,
            f"op{self.op_property.op_index}_out{i}",
        )


class PdOpDivideCodeGen:
    def __init__(
        self,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        index_program_translator_map,
    ):
        self.op_property = op_property
        self.input_properties = input_properties
        self.output_properties = output_properties
        self.kernel_arg_translator = kernel_arg_translator
        self.index_program_translator_map = index_program_translator_map

    def __call__(
        self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
    ):
        a = inputs[0]
        b = inputs[1]
        out = self.get_out_cg_val(0)
        mut_lir_code_gen_ctx.let(out, f"{a.var_name} / {b.var_name}")
        return [out]

    def get_out_cg_val(self, i):
        return code_gen_value_util.CodeGenValue(
            self.output_properties[i].type,
            f"op{self.op_property.op_index}_out{i}",
        )


class PdOpMaximumCodeGen:
    def __init__(
        self,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        index_program_translator_map,
    ):
        self.op_property = op_property
        self.input_properties = input_properties
        self.output_properties = output_properties
        self.kernel_arg_translator = kernel_arg_translator
        self.index_program_translator_map = index_program_translator_map

    def __call__(
        self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
    ):
        a = inputs[0]
        b = inputs[1]
        out = self.get_out_cg_val(0)
        mut_lir_code_gen_ctx.let(
            out,
            f"(({a.var_name} >= {b.var_name}) ? ({a.var_name}) : ({b.var_name}))",
        )
        return [out]

    def get_out_cg_val(self, i):
        return code_gen_value_util.CodeGenValue(
            self.output_properties[i].type,
            f"op{self.op_property.op_index}_out{i}",
        )


class CinnOpYieldStoreCodeGen:
    def __init__(
        self,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        index_program_translator_map,
    ):
        self.op_property = op_property
        self.input_properties = input_properties
        self.output_properties = output_properties
        self.kernel_arg_translator = kernel_arg_translator
        self.index_program_translator_map = index_program_translator_map

    def __call__(
        self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
    ):
        return inputs


class CinnOpBroadcastCodeGen:
    def __init__(
        self,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        index_program_translator_map,
    ):
        self.op_property = op_property
        self.input_properties = input_properties
        self.output_properties = output_properties
        self.kernel_arg_translator = kernel_arg_translator
        self.index_program_translator_map = index_program_translator_map

    def __call__(
        self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
    ):
        return inputs


class CinnOpExpandCodeGen:
    def __init__(
        self,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        index_program_translator_map,
    ):
        self.op_property = op_property
        self.input_properties = input_properties
        self.output_properties = output_properties
        self.kernel_arg_translator = kernel_arg_translator
        self.index_program_translator_map = index_program_translator_map

    def __call__(
        self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
    ):
        return [inputs[0]]


class CinnOpGenerateShapeCodeGen:
    def __init__(
        self,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        index_program_translator_map,
    ):
        self.op_property = op_property
        self.input_properties = input_properties
        self.output_properties = output_properties
        self.kernel_arg_translator = kernel_arg_translator
        self.index_program_translator_map = index_program_translator_map

    def __call__(
        self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx
    ):
        out = self.get_out_cg_val(0)
        return [out]

    def get_out_cg_val(self, i):
        return code_gen_value_util.CodeGenValue(
            self.output_properties[i].type,
            f"op{self.op_property.op_index}_out{i}",
        )


class OpComputeTranslatorFactory:
    def __init__(self):
        self.op_name2class = ap.OrderedDict(
            [
                ["ap_op.load_from_register", ApOpLoadFromRegisterCodeGen],
                ["ap_op.load_from_global", ApOpLoadFromGlobalCodeGen],
                ["ap_op.store_to_register", ApOpStoreToRegisterCodeGen],
                ["ap_op.store_to_global", ApOpStoreToGlobalCodeGen],
                ["pd_op.data", PdOpDataCodeGen],
                ["pd_op.full", PdOpFullCodeGen],
                ["pd_op.cast", PdOpCastCodeGen],
                ["pd_op.exp", PdOpExpCodeGen],
                ["pd_op.relu", PdOpReluCodeGen],
                ["pd_op.sin", PdOpSinCodeGen],
                ["pd_op.tanh", PdOpTanhCodeGen],
                ["pd_op.floor", PdOpFloorCodeGen],
                ["pd_op.erf", PdOpErfCodeGen],
                ["pd_op.elementwise_pow", PdOpElementwisePowCodeGen],
                ["cinn_op.scale", CinnOpScaleCodeGen],
                ["pd_op.subtract", PdOpSubtractCodeGen],
                ["pd_op.add", PdOpAddCodeGen],
                ["pd_op.multiply", PdOpMultiplyCodeGen],
                ["pd_op.divide", PdOpDivideCodeGen],
                ["pd_op.maximum", PdOpMaximumCodeGen],
                ["cinn_op.yield_store", CinnOpYieldStoreCodeGen],
                ["cinn_op.broadcast", CinnOpBroadcastCodeGen],
                ["pd_op.expand", CinnOpExpandCodeGen],
                ["cinn_op.generate_shape", CinnOpGenerateShapeCodeGen],
            ]
        )

    def __call__(
        self,
        op_property,
        input_properties,
        output_properties,
        kernel_arg_translator,
        index_program_translator_map,
    ):
        cls = self._get_class(op_property.op_name)
        return cls(
            op_property=op_property,
            input_properties=input_properties,
            output_properties=output_properties,
            kernel_arg_translator=kernel_arg_translator,
            index_program_translator_map=index_program_translator_map,
        )

    def _get_class(self, op_name):
        return self.op_name2class[op_name]
