// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <mutex>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/layout.h"

#include "paddle/phi/kernels/fusion/cutlass/int4_gemm/int4_gemm_decl.h"
#include "paddle/phi/kernels/fusion/cutlass/int4_gemm/int4_gemm_util.h"

namespace phi {
namespace fusion {
namespace cutlass_gemm_internal {
template <typename TShape,
          typename WShape,
          typename IShape,
          typename Arch,
          int NumStages>
cutlass::Status Int4GemmBiasImpl(GemmAllParams params) {
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
  using LayoutInputBias = cutlass::layout::RowMajor;
  using LayoutOutput = cutlass::layout::RowMajor;
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
  const int32_t *bias = params.bias;
  int32_t *output = params.output;
  int batch = params.batch;
  int m = params.m;
  int n = params.n;
  int k = params.k;
  cutlass::gemm::GemmCoord problem_size({m, n, k});

  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(1);

  int split_k_slices = 1;  // in big shape,no need to split k

  cutlass::layout::RowMajor layout_a(k);
  cutlass::layout::ColumnMajor layout_b(k);
  cutlass::layout::RowMajor layout_out(n);

  typename Gemm::Arguments arguments(problem_size,
                                     {input, layout_a},
                                     {weight, layout_b},
                                     {bias, layout_out},
                                     {output, layout_out},
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

// config 0
template <typename arch, int NumStages>
cutlass::Status Int4GemmBiasImpl<cutlass::gemm::GemmShape<128, 256, 128>,
                                 cutlass::gemm::GemmShape<64, 64, 128>,
                                 cutlass::gemm::GemmShape<8, 8, 32>,
                                 arch,
                                 NumStages>(GemmAllParams);
// config 1
template <typename arch, int NumStages>
cutlass::Status Int4GemmBiasImpl<cutlass::gemm::GemmShape<256, 128, 128>,
                                 cutlass::gemm::GemmShape<64, 64, 128>,
                                 cutlass::gemm::GemmShape<8, 8, 32>,
                                 arch,
                                 NumStages>(GemmAllParams);
// config 2
template <typename arch, int NumStages>
cutlass::Status Int4GemmBiasImpl<cutlass::gemm::GemmShape<128, 128, 128>,
                                 cutlass::gemm::GemmShape<64, 64, 128>,
                                 cutlass::gemm::GemmShape<8, 8, 32>,
                                 arch,
                                 NumStages>(GemmAllParams);
// config 3
template <typename arch, int NumStages>
cutlass::Status Int4GemmBiasImpl<cutlass::gemm::GemmShape<64, 128, 128>,
                                 cutlass::gemm::GemmShape<32, 64, 128>,
                                 cutlass::gemm::GemmShape<8, 8, 32>,
                                 arch,
                                 NumStages>(GemmAllParams);
// config 4
template <typename arch, int NumStages>
cutlass::Status Int4GemmBiasImpl<cutlass::gemm::GemmShape<128, 64, 128>,
                                 cutlass::gemm::GemmShape<64, 32, 128>,
                                 cutlass::gemm::GemmShape<8, 8, 32>,
                                 arch,
                                 NumStages>(GemmAllParams);
// config 5
template <typename arch, int NumStages>
cutlass::Status Int4GemmBiasImpl<cutlass::gemm::GemmShape<64, 64, 128>,
                                 cutlass::gemm::GemmShape<32, 32, 128>,
                                 cutlass::gemm::GemmShape<8, 8, 32>,
                                 arch,
                                 NumStages>(GemmAllParams);
// config 6
template <typename arch, int NumStages>
cutlass::Status Int4GemmBiasImpl<cutlass::gemm::GemmShape<256, 64, 128>,
                                 cutlass::gemm::GemmShape<64, 64, 128>,
                                 cutlass::gemm::GemmShape<8, 8, 32>,
                                 arch,
                                 NumStages>(GemmAllParams);
// config 7
template <typename arch, int NumStages>
cutlass::Status Int4GemmBiasImpl<cutlass::gemm::GemmShape<64, 256, 128>,
                                 cutlass::gemm::GemmShape<64, 64, 128>,
                                 cutlass::gemm::GemmShape<8, 8, 32>,
                                 arch,
                                 NumStages>(GemmAllParams);
// config 8
template <typename arch, int NumStages>
cutlass::Status Int4GemmBiasImpl<cutlass::gemm::GemmShape<256, 128, 128>,
                                 cutlass::gemm::GemmShape<64, 64, 128>,
                                 cutlass::gemm::GemmShape<16, 8, 64>,
                                 arch,
                                 NumStages>(GemmAllParams);
// config 9
template <typename arch, int NumStages>
cutlass::Status Int4GemmBiasImpl<cutlass::gemm::GemmShape<128, 256, 128>,
                                 cutlass::gemm::GemmShape<64, 64, 128>,
                                 cutlass::gemm::GemmShape<16, 8, 64>,
                                 arch,
                                 NumStages>(GemmAllParams);
// config 10
template <typename arch, int NumStages>
cutlass::Status Int4GemmBiasImpl<cutlass::gemm::GemmShape<256, 64, 128>,
                                 cutlass::gemm::GemmShape<64, 64, 128>,
                                 cutlass::gemm::GemmShape<16, 8, 64>,
                                 arch,
                                 NumStages>(GemmAllParams);
// config 11
template <typename arch, int NumStages>
cutlass::Status Int4GemmBiasImpl<cutlass::gemm::GemmShape<64, 256, 128>,
                                 cutlass::gemm::GemmShape<64, 64, 128>,
                                 cutlass::gemm::GemmShape<16, 8, 64>,
                                 arch,
                                 NumStages>(GemmAllParams);
// config 12
template <typename arch, int NumStages>
cutlass::Status Int4GemmBiasImpl<cutlass::gemm::GemmShape<128, 128, 128>,
                                 cutlass::gemm::GemmShape<64, 64, 128>,
                                 cutlass::gemm::GemmShape<16, 8, 64>,
                                 arch,
                                 NumStages>(GemmAllParams);
// config 13
template <typename arch, int NumStages>
cutlass::Status Int4GemmBiasImpl<cutlass::gemm::GemmShape<128, 64, 128>,
                                 cutlass::gemm::GemmShape<64, 32, 128>,
                                 cutlass::gemm::GemmShape<16, 8, 64>,
                                 arch,
                                 NumStages>(GemmAllParams);
// config 14
template <typename arch, int NumStages>
cutlass::Status Int4GemmBiasImpl<cutlass::gemm::GemmShape<64, 128, 128>,
                                 cutlass::gemm::GemmShape<32, 64, 128>,
                                 cutlass::gemm::GemmShape<16, 8, 64>,
                                 arch,
                                 NumStages>(GemmAllParams);
// config 15
template <typename arch, int NumStages>
cutlass::Status Int4GemmBiasImpl<cutlass::gemm::GemmShape<64, 64, 128>,
                                 cutlass::gemm::GemmShape<32, 32, 128>,
                                 cutlass::gemm::GemmShape<16, 8, 64>,
                                 arch,
                                 NumStages>(GemmAllParams);
// config 16
template <typename arch, int NumStages>
cutlass::Status Int4GemmBiasImpl<cutlass::gemm::GemmShape<128, 128, 256>,
                                 cutlass::gemm::GemmShape<64, 64, 256>,
                                 cutlass::gemm::GemmShape<16, 8, 64>,
                                 arch,
                                 NumStages>(GemmAllParams);
// config 17
template <typename arch, int NumStages>
cutlass::Status Int4GemmBiasImpl<cutlass::gemm::GemmShape<128, 64, 256>,
                                 cutlass::gemm::GemmShape<64, 32, 256>,
                                 cutlass::gemm::GemmShape<16, 8, 64>,
                                 arch,
                                 NumStages>(GemmAllParams);
// config 18
template <typename arch, int NumStages>
cutlass::Status Int4GemmBiasImpl<cutlass::gemm::GemmShape<64, 128, 256>,
                                 cutlass::gemm::GemmShape<32, 64, 256>,
                                 cutlass::gemm::GemmShape<16, 8, 64>,
                                 arch,
                                 NumStages>(GemmAllParams);
// config 19
template <typename arch, int NumStages>
cutlass::Status Int4GemmBiasImpl<cutlass::gemm::GemmShape<64, 64, 256>,
                                 cutlass::gemm::GemmShape<32, 32, 256>,
                                 cutlass::gemm::GemmShape<16, 8, 64>,
                                 arch,
                                 NumStages>(GemmAllParams);

std::vector<std::function<cutlass::Status(GemmAllParams)>>
    int4_gemm_bias_sm75_all_func = {
        Int4GemmBiasImpl<cutlass::gemm::GemmShape<128, 256, 128>,
                         cutlass::gemm::GemmShape<64, 64, 128>,
                         cutlass::gemm::GemmShape<8, 8, 32>,
                         cutlass::arch::Sm75,
                         2>,
        Int4GemmBiasImpl<cutlass::gemm::GemmShape<256, 128, 128>,
                         cutlass::gemm::GemmShape<64, 64, 128>,
                         cutlass::gemm::GemmShape<8, 8, 32>,
                         cutlass::arch::Sm75,
                         2>,
        Int4GemmBiasImpl<cutlass::gemm::GemmShape<128, 128, 128>,
                         cutlass::gemm::GemmShape<64, 64, 128>,
                         cutlass::gemm::GemmShape<8, 8, 32>,
                         cutlass::arch::Sm75,
                         2>,
        Int4GemmBiasImpl<cutlass::gemm::GemmShape<64, 128, 128>,
                         cutlass::gemm::GemmShape<32, 64, 128>,
                         cutlass::gemm::GemmShape<8, 8, 32>,
                         cutlass::arch::Sm75,
                         2>,
        Int4GemmBiasImpl<cutlass::gemm::GemmShape<128, 64, 128>,
                         cutlass::gemm::GemmShape<64, 32, 128>,
                         cutlass::gemm::GemmShape<8, 8, 32>,
                         cutlass::arch::Sm75,
                         2>,
        Int4GemmBiasImpl<cutlass::gemm::GemmShape<64, 64, 128>,
                         cutlass::gemm::GemmShape<32, 32, 128>,
                         cutlass::gemm::GemmShape<8, 8, 32>,
                         cutlass::arch::Sm75,
                         2>,
        Int4GemmBiasImpl<cutlass::gemm::GemmShape<256, 64, 128>,
                         cutlass::gemm::GemmShape<64, 64, 128>,
                         cutlass::gemm::GemmShape<8, 8, 32>,
                         cutlass::arch::Sm75,
                         2>,
        Int4GemmBiasImpl<cutlass::gemm::GemmShape<64, 256, 128>,
                         cutlass::gemm::GemmShape<64, 64, 128>,
                         cutlass::gemm::GemmShape<8, 8, 32>,
                         cutlass::arch::Sm75,
                         2>,
};

std::vector<std::function<cutlass::Status(GemmAllParams)>>
    int4_gemm_bias_sm80_all_func = {
        Int4GemmBiasImpl<cutlass::gemm::GemmShape<128, 256, 128>,
                         cutlass::gemm::GemmShape<64, 64, 128>,
                         cutlass::gemm::GemmShape<8, 8, 32>,
                         cutlass::arch::Sm80,
                         2>,
        Int4GemmBiasImpl<cutlass::gemm::GemmShape<256, 128, 128>,
                         cutlass::gemm::GemmShape<64, 64, 128>,
                         cutlass::gemm::GemmShape<8, 8, 32>,
                         cutlass::arch::Sm80,
                         2>,
        Int4GemmBiasImpl<cutlass::gemm::GemmShape<128, 128, 128>,
                         cutlass::gemm::GemmShape<64, 64, 128>,
                         cutlass::gemm::GemmShape<8, 8, 32>,
                         cutlass::arch::Sm80,
                         2>,
        Int4GemmBiasImpl<cutlass::gemm::GemmShape<64, 128, 128>,
                         cutlass::gemm::GemmShape<32, 64, 128>,
                         cutlass::gemm::GemmShape<8, 8, 32>,
                         cutlass::arch::Sm80,
                         2>,
        Int4GemmBiasImpl<cutlass::gemm::GemmShape<128, 64, 128>,
                         cutlass::gemm::GemmShape<64, 32, 128>,
                         cutlass::gemm::GemmShape<8, 8, 32>,
                         cutlass::arch::Sm80,
                         2>,
        Int4GemmBiasImpl<cutlass::gemm::GemmShape<64, 64, 128>,
                         cutlass::gemm::GemmShape<32, 32, 128>,
                         cutlass::gemm::GemmShape<8, 8, 32>,
                         cutlass::arch::Sm80,
                         2>,
        Int4GemmBiasImpl<cutlass::gemm::GemmShape<256, 64, 128>,
                         cutlass::gemm::GemmShape<64, 64, 128>,
                         cutlass::gemm::GemmShape<8, 8, 32>,
                         cutlass::arch::Sm80,
                         2>,
        Int4GemmBiasImpl<cutlass::gemm::GemmShape<64, 256, 128>,
                         cutlass::gemm::GemmShape<64, 64, 128>,
                         cutlass::gemm::GemmShape<8, 8, 32>,
                         cutlass::arch::Sm80,
                         2>,
        Int4GemmBiasImpl<cutlass::gemm::GemmShape<256, 128, 128>,
                         cutlass::gemm::GemmShape<64, 64, 128>,
                         cutlass::gemm::GemmShape<16, 8, 64>,
                         cutlass::arch::Sm80,
                         3>,
        Int4GemmBiasImpl<cutlass::gemm::GemmShape<128, 256, 128>,
                         cutlass::gemm::GemmShape<64, 64, 128>,
                         cutlass::gemm::GemmShape<16, 8, 64>,
                         cutlass::arch::Sm80,
                         3>,
        Int4GemmBiasImpl<cutlass::gemm::GemmShape<256, 64, 128>,
                         cutlass::gemm::GemmShape<64, 64, 128>,
                         cutlass::gemm::GemmShape<16, 8, 64>,
                         cutlass::arch::Sm80,
                         4>,
        Int4GemmBiasImpl<cutlass::gemm::GemmShape<64, 256, 128>,
                         cutlass::gemm::GemmShape<64, 64, 128>,
                         cutlass::gemm::GemmShape<16, 8, 64>,
                         cutlass::arch::Sm80,
                         4>,
        Int4GemmBiasImpl<cutlass::gemm::GemmShape<128, 128, 128>,
                         cutlass::gemm::GemmShape<64, 64, 128>,
                         cutlass::gemm::GemmShape<16, 8, 64>,
                         cutlass::arch::Sm80,
                         5>,
        Int4GemmBiasImpl<cutlass::gemm::GemmShape<128, 64, 128>,
                         cutlass::gemm::GemmShape<64, 32, 128>,
                         cutlass::gemm::GemmShape<16, 8, 64>,
                         cutlass::arch::Sm80,
                         6>,
        Int4GemmBiasImpl<cutlass::gemm::GemmShape<64, 128, 128>,
                         cutlass::gemm::GemmShape<32, 64, 128>,
                         cutlass::gemm::GemmShape<16, 8, 64>,
                         cutlass::arch::Sm80,
                         6>,
        Int4GemmBiasImpl<cutlass::gemm::GemmShape<64, 64, 128>,
                         cutlass::gemm::GemmShape<32, 32, 128>,
                         cutlass::gemm::GemmShape<16, 8, 64>,
                         cutlass::arch::Sm80,
                         10>,
        Int4GemmBiasImpl<cutlass::gemm::GemmShape<128, 128, 256>,
                         cutlass::gemm::GemmShape<64, 64, 256>,
                         cutlass::gemm::GemmShape<16, 8, 64>,
                         cutlass::arch::Sm80,
                         3>,
        Int4GemmBiasImpl<cutlass::gemm::GemmShape<128, 64, 256>,
                         cutlass::gemm::GemmShape<64, 32, 256>,
                         cutlass::gemm::GemmShape<16, 8, 64>,
                         cutlass::arch::Sm80,
                         3>,
        Int4GemmBiasImpl<cutlass::gemm::GemmShape<64, 64, 256>,
                         cutlass::gemm::GemmShape<32, 32, 256>,
                         cutlass::gemm::GemmShape<16, 8, 64>,
                         cutlass::arch::Sm80,
                         5>,
};

std::map<std::vector<int>, int> map_problem_int4_gemm_bias;
std::mutex int4_gemm_bias_mutex;

void Int4GemmBias(GemmAllParams params, int sm) {
  int batch = params.batch;
  int m = params.m;
  int n = params.n;
  int k = params.k;
  std::vector<int> problem_size = {batch, m, n, k};
  std::vector<std::function<cutlass::Status(GemmAllParams)>> *gemm_funcs =
      &int4_gemm_bias_sm75_all_func;  // default use sm75 arch
  if (sm == 80) {
    gemm_funcs = &int4_gemm_bias_sm80_all_func;
  }
  if (map_problem_int4_gemm_bias.count(problem_size)) {
    gemm_funcs->at(map_problem_int4_gemm_bias.at(problem_size))(params);
    return;
  }
  int best_config_index = ProfileToGetBestConfig(*gemm_funcs, params);
  std::lock_guard<std::mutex> guard(int4_gemm_bias_mutex);
  map_problem_int4_gemm_bias[problem_size] = best_config_index;
  gemm_funcs->at(best_config_index)(params);
}
}  // namespace cutlass_gemm_internal
}  // namespace fusion
}  // namespace phi
