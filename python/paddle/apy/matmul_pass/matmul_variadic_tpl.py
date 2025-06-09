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
import kernel_arg_translator_util
import low_level_ir_code_gen_ctx_util


def make_kernel_arg_translator():
    return kernel_arg_translator_util.KernelArgTranslator(
        param_struct_name="args"
    )


def get_anchor_iter_var_names():
    return ["coord.batch", "coord.row", "coord.column"]


class MatmulVariadicTemplate:
    def __init__(
        self,
        program_translator,
        mut_kernel_arg_id_registry,
    ):
        self.program_translator = program_translator
        self.mut_kernel_arg_id_registry = mut_kernel_arg_id_registry
        self.kernel_arg_translator = make_kernel_arg_translator()
        self.dtype2type_name = ap.OrderedDict(
            [
                [ap.PointerType.const_float_ptr, "const float*"],
                [ap.PointerType.const_float16_ptr, "const half*"],
                [ap.PointerType.const_bfloat16_ptr, "const nv_bfloat16*"],
                [ap.PointerType.float_ptr, "float*"],
                [ap.PointerType.float16_ptr, "half*"],
                [ap.PointerType.bfloat16_ptr, "nv_bfloat16*"],
                [ap.DataType.float, "float"],
                [ap.DataType.float16, "half"],
                [ap.DataType.bfloat16, "nv_bfloat16"],
                [ap.DataType.int64_t, "int64_t"],
            ]
        )
        self.input_dim_karg_to_shape_access = ap.MutableOrderedDict()
        self.kernel_name = "MatmulVariadicKernel"
        self.library_name = "matmul_variadic_kernel"

    def _register_name(self, pair):
        registry = self.mut_kernel_arg_id_registry
        registry.get_or_create_kernel_arg_id_manul_var_name(
            kernel_arg_id=pair[0], cpp_var_name=pair[1]
        )

    def compile(
        self,
        input0_karg,
        input1_karg,
        output_karg,
        input0_shape_kargs,
        input1_shape_kargs,
    ):
        kargs_name_pair_list = [
            [input0_karg, "input0"],
            [input1_karg, "input1"],
            [output_karg, "output"],
            *ap.map(
                lambda i: [input0_shape_kargs[i], f"input0_dim{i}"],
                range(len(input0_shape_kargs)),
            ),
            *ap.map(
                lambda i: [input1_shape_kargs[i], f"input1_dim{i}"],
                range(len(input1_shape_kargs)),
            ),
        ]
        print(f"-- kargs_name_pair_list: {kargs_name_pair_list}")

        ap.map(self._register_name, kargs_name_pair_list)
        mut_lir_code_gen_ctx = (
            low_level_ir_code_gen_ctx_util.CudaLikeIrCodeGenCtx(
                compute_dtype=ap.DataType.float
            )
        )

        self.program_translator.translate(
            mut_kernel_arg_id_registry=self.mut_kernel_arg_id_registry,
            mut_lir_code_gen_ctx=mut_lir_code_gen_ctx,
        )

        trivial_code_str = mut_lir_code_gen_ctx.get_stmts_joined_str(
            indent="    "
        )
        print("-- matmul_binary_epilogue_code:\n", trivial_code_str)
        project_module = self.make_project(
            trivial_code_str,
            input0_karg,
            input1_karg,
            output_karg,
            input0_shape_kargs,
            input1_shape_kargs,
        )
        return CodeGenResult(  # noqa: F821
            module=project_module,
            kernel_dispatch_func=KernelDispatch,
            kernel_dispatch_const_data=ap.SerializableAttrMap(
                kernel_args_getters=self.get_kernel_arg_runtime_getters()
            ),
        )

    def get_kernel_arg_runtime_getters(self):
        all_kernel_arg_id_and_unique_names = (
            self.mut_kernel_arg_id_registry.all_kernel_arg_id2unique_name.items()
        )
        return ap.map(
            lambda pair: pair[0].runtime_getter,
            all_kernel_arg_id_and_unique_names,
        )

    def get_kernel_arg_types(self):
        all_kernel_arg_id_and_unique_names = (
            self.mut_kernel_arg_id_registry.all_kernel_arg_id2unique_name.items()
        )
        return ap.map(
            lambda pair: pair[0].type, all_kernel_arg_id_and_unique_names
        )

    def get_kernel_arg_id_var_name(self, kernel_arg_id):
        all_kernel_arg_id2unique_name = (
            self.mut_kernel_arg_id_registry.all_kernel_arg_id2unique_name
        )
        return all_kernel_arg_id2unique_name[kernel_arg_id]

    def get_kernel_arg_list_str(self, for_declare):
        def declare_epilogue_arguments_field(pair):
            kernel_arg_id = pair[0]
            var_name = pair[1]
            field_name = self.kernel_arg_translator.get_param_struct_field_name(
                var_name
            )
            dtype = kernel_arg_id.type
            type_name = self.dtype2type_name[dtype]
            return (
                f"{type_name} {field_name}" if for_declare else f"{field_name}"
            )

        all_kernel_arg_id_and_names = (
            self.mut_kernel_arg_id_registry.all_kernel_arg_id2unique_name.items()
        )
        return ", ".join(
            ap.map(
                declare_epilogue_arguments_field, all_kernel_arg_id_and_names
            )
        )

    def get_epilogue_arguments_fields_str(self, indent):
        def declare_epilogue_arguments_field(pair):
            kernel_arg_id = pair[0]
            var_name = pair[1]
            field_name = self.kernel_arg_translator.get_param_struct_field_name(
                var_name
            )
            dtype = kernel_arg_id.type
            type_name = self.dtype2type_name[dtype]
            return f"{type_name} {field_name};"

        generated_kernel_arg_id_and_names = (
            self.mut_kernel_arg_id_registry.generated_kernel_arg_id2unique_name.items()
        )
        return f"\n{indent}".join(
            ap.map(
                declare_epilogue_arguments_field,
                generated_kernel_arg_id_and_names,
            )
        )

    def get_epilogue_arguments_init_str(self, param_obj_name, indent):
        def declare_epilogue_arguments_assign(pair):
            kernel_arg_id = pair[0]
            var_name = pair[1]
            field_name = self.kernel_arg_translator.get_param_struct_field_name(
                var_name
            )
            return f"{param_obj_name}.{field_name} = {var_name};"

        generated_kernel_arg_id_and_names = (
            self.mut_kernel_arg_id_registry.generated_kernel_arg_id2unique_name.items()
        )
        return f"\n{indent}".join(
            ap.map(
                declare_epilogue_arguments_assign,
                generated_kernel_arg_id_and_names,
            )
        )

    def get_params_input_shape_init_str(
        self, input_name, input_shape_kargs, indent
    ):
        def init_input_shape_with_args(i):
            def get_creator():
                return f"{input_name}_shape[{i}]"

            karg_var_name = self.get_kernel_arg_id_var_name(
                input_shape_kargs[i]
            )
            self.input_dim_karg_to_shape_access.get_or_create(
                karg_var_name, get_creator
            )
            return f"{indent}{input_name}_shape[{i}] = {karg_var_name};"

        shape_vector_init_str = (
            f"{input_name}_shape.resize({len(input_shape_kargs)});\n"
        )
        return shape_vector_init_str + "\n".join(
            ap.map(init_input_shape_with_args, range(len(input_shape_kargs)))
        )

    def make_project(
        self,
        trivial_code_str,
        input0_karg,
        input1_karg,
        output_karg,
        input0_shape_kargs,
        input1_shape_kargs,
    ):
        code_template = """
// auto generated codes
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>

#include "cutlass_matmul.cuh"
#include "math_function.h"
#include "profile.h"

namespace ap {

template <typename T>
struct VariadicEpilogueFunctor {
  struct Arguments {
    ${AP_EPILOGUE_ARGUMENTS_FIELDS}
  };

  // Note: need to support vectorized operation
  __forceinline__ __host__ __device__
  T operator()(T x, const Arguments& args, const MatrixCoord& coord) const {
    T out;
    ${AP_EPILOGUE_COMPUTATION_STATEMENTS}
    return out;
  }
};

template <int TuningConfigId>
static void RunMatmulWithVariadicKernel(const GemmEpilogueParams &params, ${AP_KERNEL_ARGS_DECLARE}) {
  using ElementT = ${output_dtype};
  using ElementComputeT = float;

  typename VariadicEpilogueFunctor<ElementComputeT>::Arguments epilogue_args;

  ${AP_EPILOGUE_ARGUMENTS_INIT}

  constexpr int AlignA = Alignment<ElementT, ${k_value}>::kValue;
  constexpr int AlignB = Alignment<ElementT, ${n_value}>::kValue;

  CutlassMatmulAddVariadic<ElementT, ElementComputeT, VariadicEpilogueFunctor,
                           AlignA, AlignB, TuningConfigId>(params, epilogue_args);
}

} // namespace ap

extern "C" {

void ${kernel_name}(void* stream_ptr, ${AP_KERNEL_ARGS_DECLARE}) {
  std::vector<int64_t> ${input0}_shape;
  ${AP_PARAMS_INPUT0_SHAPE_INIT}

  std::vector<int64_t> ${input1}_shape;
  ${AP_PARAMS_INPUT1_SHAPE_INIT}

  cudaStream_t* cuda_stream_ptr = reinterpret_cast<cudaStream_t*>(stream_ptr);
  ap::GemmEpilogueParams params(
      *cuda_stream_ptr, ${input0}, ${input1}, nullptr, ${output}, ${input0}_shape, ${input1}_shape, std::vector<int64_t>{});

#if AP_ENABLE_AUTOTUNE
  AP_AUTOTUNE_${output_dtype}(ap::RunMatmulWithVariadicKernel, *cuda_stream_ptr, params, ${AP_KERNEL_ARGS_CALL});
#else
  ap::RunMatmulWithVariadicKernel<ap::DefaultConfig::kConfigId>(params, ${AP_KERNEL_ARGS_CALL});
#endif
}
}
  """

        output_dtype = self.dtype2type_name[output_karg.type.data_type]
        code = (
            code_template.replace(
                "${AP_EPILOGUE_COMPUTATION_STATEMENTS}", trivial_code_str
            )
            .replace(
                "${AP_KERNEL_ARGS_DECLARE}",
                self.get_kernel_arg_list_str(for_declare=True),
            )
            .replace(
                "${AP_KERNEL_ARGS_CALL}",
                self.get_kernel_arg_list_str(for_declare=False),
            )
            .replace(
                "${AP_PARAMS_INPUT0_SHAPE_INIT}",
                self.get_params_input_shape_init_str(
                    "${input0}", input0_shape_kargs, indent="  "
                ),
            )
            .replace(
                "${AP_PARAMS_INPUT1_SHAPE_INIT}",
                self.get_params_input_shape_init_str(
                    "${input1}", input1_shape_kargs, indent="  "
                ),
            )
            .replace(
                "${AP_EPILOGUE_ARGUMENTS_FIELDS}",
                self.get_epilogue_arguments_fields_str(indent="    "),
            )
            .replace(
                "${AP_EPILOGUE_ARGUMENTS_INIT}",
                self.get_epilogue_arguments_init_str(
                    "epilogue_args", indent="  "
                ),
            )
            .replace("${kernel_name}", self.kernel_name)
            .replace("${input0}", self.get_kernel_arg_id_var_name(input0_karg))
            .replace("${input1}", self.get_kernel_arg_id_var_name(input1_karg))
            .replace("${output}", self.get_kernel_arg_id_var_name(output_karg))
            .replace("${output_dtype}", output_dtype)
            .replace("${k_value}", f"{input0_shape_kargs[-1].value}")
            .replace("${n_value}", f"{input1_shape_kargs[-1].value}")
        )

        dir_name = ap.dirname(__file__)
        source_dir = f"{dir_name}/matmul"
        cutlass_dir = f"{dir_name}/matmul/cutlass-3.7.0"
        compile_cmd = "nvcc -std=c++17 -O3 -Xcompiler=-fPIC -arch=sm_80 --expt-relaxed-constexpr"
        compile_cmd = compile_cmd + " -I " + cutlass_dir + "/include"
        compile_cmd = compile_cmd + " -I " + cutlass_dir + "/tools/util/include"
        compile_cmd = compile_cmd + " -I " + source_dir
        compile_cmd = (
            compile_cmd
            + " -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0"
        )
        compile_cmd = (
            compile_cmd + " -DAP_ENABLE_AUTOTUNE=0 -DAP_ENABLE_DEBUG=0"
        )
        compile_cmd = (
            compile_cmd
            + f" --shared {self.library_name}.cu -o lib{self.library_name}.so"
        )

        return CodeModule(  # noqa: F821
            FuncDeclare(  # noqa: F821
                ap.DataType.void,
                self.kernel_name,
                [ap.PointerType.void_ptr, *self.get_kernel_arg_types()],
            ),
            Project(  # noqa: F821
                nested_files=Project.Directory(  # noqa: F821
                    [
                        f"{self.library_name}.cu",
                        Project.FileContent(code),  # noqa: F821
                    ],
                    ["make.sh", Project.FileContent(compile_cmd)],  # noqa: F821
                ),
                compile_cmd="sh make.sh",
                so_relative_path=f"lib{self.library_name}.so",
            ),
        )


def KernelDispatch(ctx):
    import ap

    so_func = ctx.get_so_function("MatmulVariadicKernel")
    stream_ptr = ctx.device_ctx.get_stream_addr_as_void_ptr()
    getters = ctx.kernel_dispatch_const_data.kernel_args_getters
    args = [stream_ptr, *ap.map(lambda getter: getter(ctx), getters)]
    ap.apply(so_func, args)
