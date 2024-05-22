// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cutlass/gemm/device/gemm.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/gpu/gemm_fp8.h"

using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;

template <typename T>
cudaError_t CutlassSgemmNN(int M,
                           int N,
                           int K,
                           float alpha,
                           const phi::dtype::float8_e4m3fn* A,
                           int lda,
                           const phi::dtype::float8_e4m3fn * B,
                           int ldb,
                           float beta,
                           T* C,
                           int ldc,
                           size_t work_space_size,
                           const phi::GPUContext& ctx) {
  using ElementInputA =
      cutlass::float_e4m3_t;  // <- data type of elements in input matrix A
  using ElementInputB =
      cutlass::float_e4m3_t;  // <- data type of elements in input matrix B
  using ElementOutput =
      typename std::conditional_t<std::is_same_v<T, phi::dtype::bfloat16>,
                                  cutlass::bfloat16_t,
                                  cutlass::half_t>;
  using ElementAccumulator = float;

  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;

  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm90;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 64>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 64>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 32>;

  static int const kEpilogueElementsPerAccess = 1;
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        float, kEpilogueElementsPerAccess, float, float>;

  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;


  constexpr int NumStages = 2;
  int AlignmentA = 16;
  /// Access granularity of B matrix in units of elements
  int AlignmentB = 16;

  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default).
  //
  // To keep the interface manageable, several helpers are defined for plausible
  // compositions including the following example for single-precision GEMM.
  // Typical values are used as default template arguments. See
  // `cutlass/gemm/device/default_gemm_configuration.h` for more details.
  //
  // To view the full gemm device API interface, see
  // `cutlass/gemm/device/gemm.h`


  using CutlassGemm = cutlass::gemm::device::Gemm<ElementInputA,
                                               LayoutInputA,
                                               ElementInputB,
                                               LayoutInputB,
                                               ElementOutput,
                                               LayoutOutput,
                                               ElementAccumulator>;


  /*
  using CutlassGemm = cutlass::gemm::device::Gemm<ElementInputA,
                                                  LayoutInputA,
                                                  ElementInputB,
                                                  LayoutInputB,
                                                  ElementOutput,
                                                  LayoutOutput,
                                                  ElementAccumulator>;
*/
  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;

  // Construct the CUTLASS GEMM arguments object.
  //
  // One of CUTLASS's design patterns is to define gemm argument objects that
  // are constructible in host code and passed to kernels by value. These may
  // include pointers, strides, scalars, and other arguments needed by Gemm and
  // its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for
  // passing host-constructible arguments to kernels and (2.) minimized
  // initialization overhead on kernel entry.
  //
  typename CutlassGemm::Arguments args(
      {M, N, K},  // Gemm Problem dimensions
      {reinterpret_cast<ElementInputB*>(const_cast<phi::dtype::float8_e4m3fn*>(A)),
       lda},  // Tensor-ref for source matrix A
      {reinterpret_cast<ElementInputB*>(const_cast<phi::dtype::float8_e4m3fn*>(B)),
       ldb},  // Tensor-ref for source matrix B
      {reinterpret_cast<ElementOutput*>(C),
       ldc},  // Tensor-ref for source matrix C
      {reinterpret_cast<ElementOutput*>(C),
       ldc},  // Tensor-ref for destination matrix D (may be different memory
              // than source C matrix)
      {alpha, beta});  // Scalars used in the Epilogue

  CutlassGemm gemm;
  auto can_implement = gemm.can_implement(args);
  if (can_implement != cutlass::Status::kSuccess) {
    std::string err_msg =
        "fpA_intB cutlass kernel will fail for params. Error: " +
        std::string(cutlassGetStatusString(can_implement));
    throw std::runtime_error("[fpA_intB Runner] " + err_msg);
  }

  auto workspace = phi::memory_utils::Alloc(ctx.GetPlace(), gemm.get_workspace_size(args));

  auto init_status =
      gemm.initialize(args, workspace->ptr(), ctx.stream());
  if (init_status != cutlass::Status::kSuccess) {
    std::string err_msg =
        "Failed to initialize cutlass fpA_intB gemm. Error: " +
        std::string(cutlassGetStatusString(init_status));
    throw std::runtime_error("[fpA_intB Runner] " + err_msg);
  }

  auto run_status = gemm.run(ctx.stream());

  if (run_status != cutlass::Status::kSuccess) {
    std::string err_msg = "Failed to run cutlass fpA_intB gemm. Error: " +
                          std::string(cutlassGetStatusString(run_status));
    throw std::runtime_error("[fpA_intB Runner] " + err_msg);
  }
  //
  // Launch the CUTLASS GEMM kernel.
  //

  // cutlass::Status status = gemm_operator(args);

  //
  // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
  //

  if (run_status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}



template <typename T>
void CublasLtMatmulFP8(const phi::GPUContext& dev_ctx,
                       const phi::DenseTensor& mat_a,
                       const phi::DenseTensor& mat_b,
                       phi::DenseTensor* workspace,
                       phi::DenseTensor* out) {
  size_t work_space_size = workspace->numel();
  std::cout << "test::::CutlasssMatmulFP8 " << mat_a.dims()[0] << std::endl;
  std::cout << "test::::CutlasssMatmulFP8 " << work_space_size << std::endl;
  int m = mat_a.dims()[0];
  int k = mat_a.dims()[1];
  int n = mat_b.dims()[1];

  float alpha_ = 1.0f;
  float beta_ = 0.0f;

  CutlassSgemmNN<T>(m,
                    n,
                    k,
                    alpha_,
                    mat_a.data<phi::dtype::float8_e4m3fn>(),
                    m,
                    mat_b.data<phi::dtype::float8_e4m3fn>(),
                    k,
                    beta_,
                    out->data<T>(),
                    m,
                    work_space_size,
                    dev_ctx);
}

template void CublasLtMatmulFP8<phi::dtype::float16>(const phi::GPUContext& dev_ctx,
                       const phi::DenseTensor& mat_a,
                       const phi::DenseTensor& mat_b,
                       phi::DenseTensor* workspace,
                       phi::DenseTensor* out);

template void CublasLtMatmulFP8<phi::dtype::bfloat16>(const phi::GPUContext& dev_ctx,
                       const phi::DenseTensor& mat_a,
                       const phi::DenseTensor& mat_b,
                       phi::DenseTensor* workspace,
                       phi::DenseTensor* out);
