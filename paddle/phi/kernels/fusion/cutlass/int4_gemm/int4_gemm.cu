// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "paddle/phi/kernels/fusion/cutlass/int4_gemm/int4_gemm_decl.h"
#include "paddle/phi/kernels/fusion/cutlass/int4_gemm/int4_gemm_util.h"

namespace phi {
namespace fusion {
namespace cutlass_gemm_internal {
template <typename TShape,
          typename WShape,
          typename IShape,
          typename Arch,
          int NumStages = 2>
cutlass::Status Int4Gemm(GemmAllParams params) {
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = int32_t;
  using ElementOutput = int32_t;
  using ElementInputA = cutlass::int4b_t;
  using ElementInputB = cutlass::int4b_t;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = Arch;
  using ThreadblockShape = TShape;
  using WarpShape = WShape;
  using InstructionShape = IShape;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementComputeEpilogue>;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                           LayoutInputA,
                                           ElementInputB,
                                           LayoutInputB,
                                           ElementOutput,
                                           LayoutOutput,
                                           ElementAccumulator,
                                           MMAOp,
                                           SmArch,
                                           ThreadblockShape,
                                           WarpShape,
                                           InstructionShape,
                                           EpilogueOp,
                                           SwizzleThreadBlock,
                                           NumStages>;
  const cutlass::int4b_t *input = params.input;
  const cutlass::int4b_t *weight = params.weight;
  const cutlass::int4b_t *bias = params.bias;
  cutlass::int4b_t *output = params.output;
  int batch = params.batch;
  int m = params.m;
  int n = params.n;
  int k = params.k;
  cutlass::gemm::GemmCoord problem_size({m, n, k});

  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);

  int split_k_slices = 1;  // in big shape,no need to split k

  typename Gemm::Arguments arguments(problem_size,
                                     {input, {}},
                                     {weight, {}},
                                     {bias, {}},
                                     {output, {}},
                                     {alpha, beta},
                                     split_k_slices);

  Gemm gemm_op;
  size_t workspace_size = gemm_op.get_workspace_size(arguments);
  auto ctx = params.ctx;
  auto stream = ctx->stream();
  auto tmp_gpu_ptrs_data = paddle::memory::Alloc(
      ctx->GetPlace(),
      workspace_size,
      phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  void *workspace = tmp_gpu_ptrs_data->ptr();
  auto status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace);
  CUTLASS_CHECK(status);
  status = gemm_op(stream);
  CUTLASS_CHECK(status);
  return status;
}
}  // namespace cutlass_gemm_internal
}  // namespace fusion
}  // namespace phi
