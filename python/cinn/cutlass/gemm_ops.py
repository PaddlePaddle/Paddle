# Copyright (c) 2023 CINN Authors. All Rights Reserved.
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
from .library import (
    DataType,
    DataTypeNames,
    DataTypeSize,
    DataTypeTag,
    EpilogueFunctor,
    EpilogueFunctorTag,
    LayoutTag,
    LayoutType,
    MathOperationTag,
    OpcodeClass,
    OpcodeClassNames,
    OpcodeClassTag,
    OperationKind,
    ShortDataTypeNames,
    ShortLayoutTypeNames,
    SwizzlingFunctor,
    SwizzlingFunctorTag,
    TensorDescription,
    substitute_template,
)

MATMUL_EPILOGUE_MAP = {
    "cutlass_matmul": (EpilogueFunctor.LinearCombination, False),
    "cutlass_matmul_bias": (EpilogueFunctor.LinearCombinationBias, True),
    "cutlass_matmul_bias_relu": (EpilogueFunctor.LinearCombinationRelu, True),
    "cutlass_matmul_bias_gelu_fp16": (
        EpilogueFunctor.LinearCombinationGelu,
        False,
    ),
    "cutlass_matmul_bias_gelu_fp32": (
        EpilogueFunctor.LinearCombinationGelu,
        False,
    ),
    "cutlass_batch_matmul": (EpilogueFunctor.LinearCombination, False),
}


class GemmProfilerEmitter:
    """Emit a C++ source for profiling CUTLASS kernels."""

    def __init__(self):
        from jinja2 import Template

        self.template = Template(
            """
#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>

#include "cuda_runtime.h"
#include "cutlass/gemm/device/gemm.h"

#define CUTLASS_CHECK(status)                                                                    \\
  {                                                                                              \\
    cutlass::Status error = status;                                                              \\
    if (error != cutlass::Status::kSuccess) {                                                    \\
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \\
                << std::endl;                                                                    \\
      exit(EXIT_FAILURE);                                                                        \\
    }                                                                                            \\
  }

#define CUDA_CHECK(status)                                              \\
  {                                                                     \\
    cudaError_t error = status;                                         \\
    if (error != cudaSuccess) {                                         \\
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \\
                << " at line: " << __LINE__ << std::endl;               \\
      exit(EXIT_FAILURE);                                               \\
    }                                                                   \\
  }

template<typename DTypeA, typename DTypeB, typename DTypeC>
cudaError_t CutlassGemmRCR(
    int M,
    int N,
    int K,
    DTypeC alpha,
    DTypeA const *A,
    int lda,
    DTypeB const *B,
    int ldb,
    DTypeC beta,
    DTypeC *C,
    int ldc) {
  using namespace std::chrono;
  {{OperatorDef}}
  Operation_{{OperatorName}} gemm_operator;
  Operation_{{OperatorName}}::Arguments args({M, N, K},
                              {A, lda},
                              {B, ldb},
                              {C, ldc},
                              {C, ldc},
                              {alpha, beta});
  cutlass::Status status = gemm_operator(args);
  CUTLASS_CHECK(status)

  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  for (int i = 0; i < 100; ++i) {
    status = gemm_operator(args);
  }
  cudaDeviceSynchronize();
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
  std::cout << time_span.count() << std::endl;
  return cudaSuccess;
}


template<typename DType>
cudaError_t AllocateMatrix(DType **matrix, int ldm, int rows, int columns, int seed = 0) {
  cudaError_t result;

  size_t sizeof_matrix = sizeof(DType) * rows * columns;

  // Allocate device memory.
  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Clear the allocation.
  result = cudaMemset(*matrix, 0, sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to clear matrix device memory: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  if (result != cudaSuccess) {
    std::cerr << "Failed to initialize matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  return result;
}

template<typename DTypeA, typename DTypeB, typename DTypeC>
cudaError_t TestCutlassGemm(int M, int N, int K, DTypeC alpha, DTypeC beta) {
  cudaError_t result;

  {{LeadingDim}}
  // size_t sizeof_C = sizeof(DTypeC) * ldc * N;
  DTypeA *A;
  DTypeB *B;
  DTypeC *C_cutlass;
  result = AllocateMatrix<DTypeA>(&A, lda, M, K, 0);
  if (result !=  cudaSuccess) {
    return result;
  }
  result = AllocateMatrix<DTypeB>(&B, ldb, K, N, 17);
  if (result !=  cudaSuccess) {
    cudaFree(A);
    return result;
  }
  result = AllocateMatrix<DTypeC>(&C_cutlass, ldc, M, N, 101);
  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    return result;
  }
  result = CutlassGemmRCR<DTypeA, DTypeB, DTypeC>(M, N, K, alpha, A, lda, B, ldb,
                                                  beta, C_cutlass, ldc);
  if (result != cudaSuccess) {
    std::cerr << "CUTLASS GEMM kernel failed: "
      << cudaGetErrorString(result) << std::endl;
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }
  cudaFree(C_cutlass);
  cudaFree(B);
  cudaFree(A);
  return cudaSuccess;
}

int main(int argc, const char *arg[]) {
  int problem[3] = { 4096, 4096, 4096 };
  for (int i = 1; i < argc && i < 4; ++i) {
    std::stringstream ss(arg[i]);
    ss >> problem[i - 1];
  }
  float scalars[2] = { 1, 0 };
  cudaError_t result = TestCutlassGemm< {{DTypeA}}, {{DTypeB}}, {{DTypeC}}>(
    problem[0],     // GEMM M dimension
    problem[1],     // GEMM N dimension
    problem[2],     // GEMM K dimension
    static_cast<{{DTypeC}}>(scalars[0]),     // alpha
    static_cast<{{DTypeC}}>(scalars[1])      // beta
  );
  return result == cudaSuccess ? 0 : -1;
}
"""
        )

    def emit(self, op_name, op_def, dtype_a, dtype_b, dtype_c, ld):
        src = self.template.render(
            OperatorName=op_name,
            OperatorDef=op_def,
            DTypeA=dtype_a,
            DTypeB=dtype_b,
            DTypeC=dtype_c,
            LeadingDim=ld,
        )
        return src


class GemmOperation:
    """Describes various attributes for instantiating GEMM kernels."""

    def __init__(
        self,
        arch,
        tile_description,
        A,
        B,
        C,
        element_epilogue,
        epilogue_functor=EpilogueFunctor.LinearCombination,
        swizzling_functor=SwizzlingFunctor.Identity8,
    ):
        self.operation_kind = OperationKind.Gemm
        self.arch = arch
        self.tile_description = tile_description
        self.A = A
        self.B = B
        self.C = C
        self.element_epilogue = element_epilogue
        self.epilogue_functor = epilogue_functor
        self.swizzling_functor = swizzling_functor

    def accumulator_type(self):
        return self.tile_description.math_instruction.element_accumulator

    def short_math_name(self):
        return ShortDataTypeNames[self.accumulator_type()]

    def core_name(self):
        """The basic operation kind is prefixed with a letter indicating the accumulation type."""
        inst_shape = ""
        intermediate_type = ""

        if (
            self.tile_description.math_instruction.opcode_class
            == OpcodeClass.TensorOp
            or self.tile_description.math_instruction.opcode_class
            == OpcodeClass.WmmaTensorOp
        ):
            inst_shape = "%d%d%d" % tuple(
                self.tile_description.math_instruction.instruction_shape
            )
            if (
                self.tile_description.math_instruction.element_a
                != self.A.element
                and self.tile_description.math_instruction.element_a
                != self.tile_description.math_instruction.element_accumulator
            ):
                intermediate_type = DataTypeNames[
                    self.tile_description.math_instruction.element_a
                ]

        return f"{self.short_math_name()}{inst_shape}{intermediate_type}gemm"

    def extended_name(self):
        """Append data types if they differ from compute type."""
        if (
            self.C.element
            != self.tile_description.math_instruction.element_accumulator
            and self.A.element
            != self.tile_description.math_instruction.element_accumulator
        ):
            extended_name = "${element_c}_${core_name}_${element_a}"
        elif (
            self.C.element
            == self.tile_description.math_instruction.element_accumulator
            and self.A.element
            != self.tile_description.math_instruction.element_accumulator
        ):
            extended_name = "${core_name}_${element_a}"
        else:
            extended_name = "${core_name}"

        extended_name = substitute_template(
            extended_name,
            {
                "element_a": DataTypeNames[self.A.element],
                "element_c": DataTypeNames[self.C.element],
                "core_name": self.core_name(),
            },
        )

        return extended_name

    def layout_name(self):
        return f"{ShortLayoutTypeNames[self.A.layout]}{ShortLayoutTypeNames[self.B.layout]}"

    def procedural_name(self):
        """The full procedural name indicates architecture, extended name, tile size,
        and layout.
        """
        threadblock = self.tile_description.procedural_name()
        opcode_class_name = OpcodeClassNames[
            self.tile_description.math_instruction.opcode_class
        ]

        return substitute_template(
            "cutlass_${opcode_class}_${extended_name}_${threadblock}_${layout}_align${alignment}",
            {
                "opcode_class": opcode_class_name,
                "extended_name": self.extended_name(),
                "threadblock": threadblock,
                "layout": self.layout_name(),
                "alignment": f"{self.A.alignment}",
            },
        )

    def leading_dim(self):
        """lda, ldb, ldc, according to the leading dimension."""
        if self.A.layout == LayoutType.RowMajor:
            lda = "K"
        elif self.A.layout == LayoutType.ColumnMajor:
            lda = "M"
        else:
            ValueError("The layout of A is not implemented.")

        if self.B.layout == LayoutType.RowMajor:
            ldb = "N"
        elif self.B.layout == LayoutType.ColumnMajor:
            ldb = "K"
        else:
            ValueError("The layout of B is not implemented.")

        if self.C.layout == LayoutType.RowMajor:
            ldc = "N"
        elif self.C.layout == LayoutType.ColumnMajor:
            ldc = "M"
        else:
            ValueError("The layout of B is not implemented.")

        return substitute_template(
            "int lda = ${lda_val};\n\tint ldb = ${ldb_val};\n\tint ldc = ${ldc_val};\n",
            {"lda_val": lda, "ldb_val": ldb, "ldc_val": ldc},
        )


class EmitGemmInstance:
    """Responsible for emitting a CUTLASS template definition."""

    def __init__(self):
        self.epilogue_default = """
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >"""
        self.epilogue_no_beta_scaling = """
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue},
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >"""
        self.gemm_template = """
  // Gemm operator ${operation_name}
  using Operation_${operation_name} = cutlass::gemm::device::${kernel_name}<
    ${element_a}, ${layout_a},
    ${element_b}, ${layout_b},
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue},
    ${swizzling_functor},
    ${stages},
    ${align_a},
    ${align_b},
    ${split_k_serial}
    ${math_operation}
  >;
"""

    def emit(self, operation, no_beta_scaling=False, batched=False):
        """Instantiate a GEMM kernel from given `operation`."""
        warp_shape = [
            operation.tile_description.threadblock_shape[idx]
            // operation.tile_description.warp_count[idx]
            for idx in range(3)
        ]
        epilogue_vector_length = (
            min(operation.C.alignment * DataTypeSize[operation.C.element], 128)
            // DataTypeSize[operation.C.element]
        )
        values = {
            "operation_name": operation.procedural_name(),
            "element_a": DataTypeTag[operation.A.element],
            "layout_a": LayoutTag[operation.A.layout],
            "element_b": DataTypeTag[operation.B.element],
            "layout_b": LayoutTag[operation.B.layout],
            "element_c": DataTypeTag[operation.C.element],
            "layout_c": LayoutTag[operation.C.layout],
            "element_accumulator": DataTypeTag[operation.accumulator_type()],
            "opcode_class": OpcodeClassTag[
                operation.tile_description.math_instruction.opcode_class
            ],
            "arch": f"cutlass::arch::Sm{operation.arch}",
            "threadblock_shape_m": str(
                operation.tile_description.threadblock_shape[0]
            ),
            "threadblock_shape_n": str(
                operation.tile_description.threadblock_shape[1]
            ),
            "threadblock_shape_k": str(
                operation.tile_description.threadblock_shape[2]
            ),
            "warp_shape_m": str(warp_shape[0]),
            "warp_shape_n": str(warp_shape[1]),
            "warp_shape_k": str(warp_shape[2]),
            "instruction_shape_m": str(
                operation.tile_description.math_instruction.instruction_shape[0]
            ),
            "instruction_shape_n": str(
                operation.tile_description.math_instruction.instruction_shape[1]
            ),
            "instruction_shape_k": str(
                operation.tile_description.math_instruction.instruction_shape[2]
            ),
            "epilogue_vector_length": str(epilogue_vector_length),
            "element_epilogue": str(DataTypeTag[operation.element_epilogue]),
            "epilogue_functor": EpilogueFunctorTag[operation.epilogue_functor],
            "swizzling_functor": SwizzlingFunctorTag[
                operation.swizzling_functor
            ],
            "stages": str(operation.tile_description.stages),
            "align_a": str(operation.A.alignment),
            "align_b": str(operation.B.alignment),
            "math_operation": MathOperationTag[
                operation.tile_description.math_instruction.math_operation
            ],
        }

        values["kernel_name"] = "GemmBatched" if batched else "Gemm"
        values["split_k_serial"] = "" if batched else "false,"

        gemm_template = substitute_template(
            self.gemm_template,
            {
                "epilogue": self.epilogue_no_beta_scaling
                if no_beta_scaling
                else self.epilogue_default
            },
        )
        return substitute_template(gemm_template, values)


def instantiate_gemm_template(attrs, func_args):
    """Return CUTLASS host code for GEMM based on a template and the provided attribute map."""

    template = """
  using ElementInputA = ${ElementInputA};
  using ElementInputB = ${ElementInputB};
  using ElementOutput = ${ElementOutput};
  using ElementComputeEpilogue = ${ElementOutput};

  ${cutlass_op_def}

  using ${kernel} = Operation_${cutlass_op_name};
  int M = ${M};
  int N = ${N};
  int K = ${K};
  cutlass::gemm::GemmCoord problem_size(M, N, K);
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(${beta});
  void* ptr_a = (void*)(${arg0}->data);
  void* ptr_b = (void*)(${arg1}->data);
  ${bias_decl}
  void* ptr_out = (void*)(out0->data);

  typename ${kernel}::Arguments arguments{
   problem_size,
   {static_cast<ElementInputA*>(ptr_a), ${lda}}, ${batch_stride_A}
   {static_cast<ElementInputB*>(ptr_b), ${ldb}}, ${batch_stride_B}
   {static_cast<ElementOutput*>(${ptr_c}), ${c_stride}}, ${batch_stride_C}
   {static_cast<ElementOutput*>(ptr_out), ${ldc}}, ${batch_stride_C}
   {${alpha_beta}},
   ${split_k_slices_or_batch}
  };
  size_t workspace_size = ${kernel}::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  ${kernel} gemm_op;
  cutlass::Status status = gemm_op.can_implement(arguments);
  CHECK(status == cutlass::Status::kSuccess);
  status = gemm_op.initialize(arguments, workspace.get());
  CHECK(status == cutlass::Status::kSuccess);
  status = gemm_op();
  CHECK(status == cutlass::Status::kSuccess);
"""
    has_bias = "bias" in attrs["op_type"]
    is_gelu = "gelu" in attrs["op_type"]
    batched = "batch_matmul" in attrs["op_type"]

    aux_map = {"kernel": "Gemm"}

    if has_bias:
        aux_map.update(
            {
                "bias_decl": "void* ptr_c_bias = (void*)(${arg2}->data);\n",
                "ptr_c": "ptr_c_bias",
                "c_stride": "0",
            }
        )
    else:
        aux_map.update(
            {"bias_decl": "", "ptr_c": "ptr_out", "c_stride": attrs["ldc"]}
        )

    if is_gelu:
        # GeLU epilogue does not compile with NoBetaScaling, so we explicitly specify the scale.
        aux_map["beta"] = "1"
    else:
        aux_map["beta"] = "0"

    if has_bias and not is_gelu:
        aux_map["alpha_beta"] = "alpha"
    else:
        aux_map["alpha_beta"] = "alpha, beta"

    for key in ["batch_stride_A", "batch_stride_B", "batch_stride_C"]:
        if not batched:
            aux_map[key] = ""
        else:
            aux_map[key] = attrs[key] + ","

    if batched:
        attrs["split_k_slices_or_batch"] = attrs["batch"]
    else:
        attrs["split_k_slices_or_batch"] = "1"

    template = substitute_template(template, aux_map)

    for i, arg in enumerate(func_args):
        attrs[f"arg{i}"] = arg

    return substitute_template(template, attrs)


def create_gemm_operator_with_epilogue(
    op_type,
    tile_description,
    data_type,
    alignment,
    swizzling_functor,
    batched=False,
):
    """
    Instantiate a cutlass kernel from the given configuration,
    along with the epilouge functor
    """
    element_a, element_b, element_c, element_epilogue = data_type

    A = TensorDescription(element_a, LayoutType.RowMajor, alignment)
    B = TensorDescription(element_b, LayoutType.ColumnMajor, alignment)
    C = TensorDescription(element_c, LayoutType.RowMajor, alignment)

    if batched:
        swizzling_functor = SwizzlingFunctor.Batched

    epilogue, no_beta_scaling = MATMUL_EPILOGUE_MAP[op_type]

    op = GemmOperation(
        tile_description.minimum_compute_capability,
        tile_description,
        A,
        B,
        C,
        element_epilogue,
        epilogue,
        swizzling_functor,
    )

    return (
        op.procedural_name(),
        EmitGemmInstance().emit(
            op, no_beta_scaling=no_beta_scaling, batched=batched
        ),
    )


def enumerate_gemm_operators(
    tile_descriptions,
    data_type,
    alignment_constraints,
    swizzling_functor=SwizzlingFunctor.Identity8,
):
    """Exhaustively instantiate all kernels from a given configuration."""
    ret = []
    kernel_emitter = EmitGemmInstance()
    profiler_emitter = GemmProfilerEmitter()

    element_a, element_b, element_c, element_epilogue = data_type

    for tile_description in tile_descriptions:
        for alignment in alignment_constraints:
            A = TensorDescription(element_a, LayoutType.RowMajor, alignment)
            B = TensorDescription(element_b, LayoutType.ColumnMajor, alignment)
            C = TensorDescription(element_c, LayoutType.RowMajor, alignment)

            if element_c == DataType.s32 and A.alignment == 1:
                tile_description.threadblock_shape[0] = min(
                    tile_description.threadblock_shape[0], 128
                )
                tile_description.threadblock_shape[1] = min(
                    tile_description.threadblock_shape[1], 128
                )

            op = GemmOperation(
                tile_description.minimum_compute_capability,
                tile_description,
                A,
                B,
                C,
                element_epilogue,
                EpilogueFunctor.LinearCombination,
                swizzling_functor,
            )

            src = profiler_emitter.emit(
                op.procedural_name(),
                kernel_emitter.emit(op, batched=False),
                DataTypeTag[element_a],
                DataTypeTag[element_b],
                DataTypeTag[element_c],
                op.leading_dim(),
            )

            ret.append(
                {
                    "src": src,
                    "op": op,
                    "name": op.procedural_name(),
                    "tile_description": tile_description,
                    "alignment": alignment,
                    "data_type": data_type,
                    "swizzle_functor": swizzling_functor,
                }
            )

    return ret
