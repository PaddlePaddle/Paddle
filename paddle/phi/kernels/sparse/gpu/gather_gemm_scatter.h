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
#pragma once

#ifdef PADDLE_WITH_CUTLASS
#include "cutlass/arch/mma.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/util/device_memory.h"
#include "examples/common/helper.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
namespace phi {
namespace sparse {
typedef void (*fp16_gather_gemm_scatter)(const GPUContext& dev_ctx,
                                         const cutlass::half_t* const a,
                                         const cutlass::half_t* const b,
                                         const cutlass::half_t* const c,
                                         cutlass::half_t* const d,
                                         const int m,
                                         const int n,
                                         const int k,
                                         const int32_t* a_indices,
                                         const int32_t* c_d_indices,
                                         cutlass::half_t const alpha,
                                         cutlass::half_t const beta);
typedef void (*fp32_gather_gemm_scatter)(const GPUContext& dev_ctx,
                                         const float* const a,
                                         const float* const b,
                                         const float* const c,
                                         float* const d,
                                         const int m,
                                         const int n,
                                         const int k,
                                         const int32_t* a_indices,
                                         const int32_t* c_d_indices,
                                         float const alpha,
                                         float const beta);
typedef void (*fp64_gather_gemm_scatter)(const GPUContext& dev_ctx,
                                         const double* const a,
                                         const double* const b,
                                         const double* const c,
                                         double* const d,
                                         const int m,
                                         const int n,
                                         const int k,
                                         const int32_t* a_indices,
                                         const int32_t* c_d_indices,
                                         double const alpha,
                                         double const beta);
fp16_gather_gemm_scatter getBestFp16Kernel(const int M,
                                           const int K,
                                           const int N);
fp32_gather_gemm_scatter getBestFp32Kernel(const int M,
                                           const int K,
                                           const int N,
                                           const int SM);
fp64_gather_gemm_scatter getBestFp64Kernel(const int M,
                                           const int K,
                                           const int N);
template <typename T, typename Gemm>
void launchKernel(const GPUContext& dev_ctx,
                  const T* const a,
                  const T* const b,
                  const T* const c,
                  T* const d,
                  const int m,
                  const int n,
                  const int k,
                  const int32_t* a_indices,
                  const int32_t* c_d_indices,
                  T const alpha,
                  T const beta) {
  cutlass::gemm::GemmCoord problem_size_real({m, n, k});
  int split_k_slices = 1;
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size_real,
      split_k_slices,
      {alpha, beta},
      a,
      b,
      c,
      d,
      cutlass::layout::RowMajor().capacity(problem_size_real.mk()),
      cutlass::layout::RowMajor().capacity(problem_size_real.kn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      cutlass::layout::RowMajor().capacity(problem_size_real.mn()),
      problem_size_real.k(),
      problem_size_real.n(),
      problem_size_real.n(),
      problem_size_real.n(),
      a_indices,
      nullptr,
      c_d_indices};
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  Gemm gemm_op;
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  gemm_op(dev_ctx.stream());
}
struct cutlass_tensorop_h1688gemm_128x64_32x2_nn_align8 {
  using Gemm = cutlass::gemm::device::GemmUniversal<
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75,
      cutlass::gemm::GemmShape<128, 64, 32>,
      cutlass::gemm::GemmShape<64, 32, 32>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t,
                                                   8,
                                                   cutlass::half_t,
                                                   cutlass::half_t>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      2,
      8,
      8,
      cutlass::arch::OpMultiplyAdd,
      cutlass::ComplexTransform::kNone,
      cutlass::ComplexTransform::kNone,
      true,
      false,
      true>;
};
struct cutlass_tensorop_h1688gemm_64x128_32x2_nn_align8 {
  using Gemm = cutlass::gemm::device::GemmUniversal<
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75,
      cutlass::gemm::GemmShape<64, 128, 32>,
      cutlass::gemm::GemmShape<32, 64, 32>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t,
                                                   8,
                                                   cutlass::half_t,
                                                   cutlass::half_t>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      2,
      8,
      8,
      cutlass::arch::OpMultiplyAdd,
      cutlass::ComplexTransform::kNone,
      cutlass::ComplexTransform::kNone,
      true,
      false,
      true>;
};
struct cutlass_tensorop_h1688gemm_128x64_32x2_nn_align4 {
  using Gemm = cutlass::gemm::device::GemmUniversal<
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75,
      cutlass::gemm::GemmShape<128, 64, 32>,
      cutlass::gemm::GemmShape<64, 32, 32>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t,
                                                   4,
                                                   cutlass::half_t,
                                                   cutlass::half_t>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      2,
      4,
      4,
      cutlass::arch::OpMultiplyAdd,
      cutlass::ComplexTransform::kNone,
      cutlass::ComplexTransform::kNone,
      true,
      false,
      true>;
};
struct cutlass_tensorop_h1688gemm_64x64_32x2_nn_align4 {
  using Gemm = cutlass::gemm::device::GemmUniversal<
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<32, 32, 32>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t,
                                                   4,
                                                   cutlass::half_t,
                                                   cutlass::half_t>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      2,
      4,
      4,
      cutlass::arch::OpMultiplyAdd,
      cutlass::ComplexTransform::kNone,
      cutlass::ComplexTransform::kNone,
      true,
      false,
      true>;
};
struct cutlass_tensorop_h1688gemm_64x64_32x2_nn_align8 {
  using Gemm = cutlass::gemm::device::GemmUniversal<
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<32, 32, 32>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t,
                                                   8,
                                                   cutlass::half_t,
                                                   cutlass::half_t>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      2,
      8,
      8,
      cutlass::arch::OpMultiplyAdd,
      cutlass::ComplexTransform::kNone,
      cutlass::ComplexTransform::kNone,
      true,
      false,
      true>;
};
struct cutlass_tensorop_h16816gemm_64x64_64x5_nn_align8 {
  using Gemm = cutlass::gemm::device::GemmUniversal<
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 64, 64>,
      cutlass::gemm::GemmShape<32, 32, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t,
                                                   8,
                                                   cutlass::half_t,
                                                   cutlass::half_t>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      5,
      8,
      8,
      cutlass::arch::OpMultiplyAdd,
      cutlass::ComplexTransform::kNone,
      cutlass::ComplexTransform::kNone,
      true,
      false,
      true>;
};
struct cutlass_tensorop_f16_s1688gemm_f16_64x128_32x2_nn_align8 {
  using Gemm = cutlass::gemm::device::GemmUniversal<
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75,
      cutlass::gemm::GemmShape<64, 128, 32>,
      cutlass::gemm::GemmShape<32, 64, 32>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::
          LinearCombination<cutlass::half_t, 8, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      2,
      8,
      8,
      cutlass::arch::OpMultiplyAdd,
      cutlass::ComplexTransform::kNone,
      cutlass::ComplexTransform::kNone,
      true,
      false,
      true>;
};
struct cutlass_tensorop_f16_s1688gemm_f16_64x64_32x2_nn_align8 {
  using Gemm = cutlass::gemm::device::GemmUniversal<
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<32, 32, 32>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::
          LinearCombination<cutlass::half_t, 8, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      2,
      8,
      8,
      cutlass::arch::OpMultiplyAdd,
      cutlass::ComplexTransform::kNone,
      cutlass::ComplexTransform::kNone,
      true,
      false,
      true>;
};
struct cutlass_tensorop_s1688f16gemm_64x64_16x10_nn_align4 {
  using Gemm = cutlass::gemm::device::GemmUniversal<
      float,
      cutlass::layout::RowMajor,
      float,
      cutlass::layout::RowMajor,
      float,
      cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 64, 16>,
      cutlass::gemm::GemmShape<32, 32, 16>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      10,
      4,
      4,
      cutlass::arch::OpMultiplyAddFastF16,
      cutlass::ComplexTransform::kNone,
      cutlass::ComplexTransform::kNone,
      true,
      false,
      true>;
};
struct cutlass_tensorop_s1688f16gemm_128x128_16x3_nn_align4 {
  using Gemm = cutlass::gemm::device::GemmUniversal<
      float,
      cutlass::layout::RowMajor,
      float,
      cutlass::layout::RowMajor,
      float,
      cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 16>,
      cutlass::gemm::GemmShape<64, 64, 16>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      3,
      4,
      4,
      cutlass::arch::OpMultiplyAddFastF16,
      cutlass::ComplexTransform::kNone,
      cutlass::ComplexTransform::kNone,
      true,
      false,
      true>;
};
struct cutlass_tensorop_s1688f16gemm_256x64_16x4_nn_align4 {
  using Gemm = cutlass::gemm::device::GemmUniversal<
      float,
      cutlass::layout::RowMajor,
      float,
      cutlass::layout::RowMajor,
      float,
      cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 64, 16>,
      cutlass::gemm::GemmShape<64, 64, 16>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      4,
      4,
      4,
      cutlass::arch::OpMultiplyAddFastF16,
      cutlass::ComplexTransform::kNone,
      cutlass::ComplexTransform::kNone,
      true,
      false,
      true>;
};
struct cutlass_tensorop_s1688tf32gemm_256x128_16x3_nn_align4 {
  using Gemm = cutlass::gemm::device::GemmUniversal<
      float,
      cutlass::layout::RowMajor,
      float,
      cutlass::layout::RowMajor,
      float,
      cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 128, 16>,
      cutlass::gemm::GemmShape<64, 64, 16>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      3,
      4,
      4,
      cutlass::arch::OpMultiplyAdd,
      cutlass::ComplexTransform::kNone,
      cutlass::ComplexTransform::kNone,
      true,
      false,
      true>;
};
struct cutlass_tensorop_s1688f16gemm_64x128_16x6_nn_align4 {
  using Gemm = cutlass::gemm::device::GemmUniversal<
      float,
      cutlass::layout::RowMajor,
      float,
      cutlass::layout::RowMajor,
      float,
      cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 128, 16>,
      cutlass::gemm::GemmShape<32, 64, 16>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      6,
      4,
      4,
      cutlass::arch::OpMultiplyAddFastF16,
      cutlass::ComplexTransform::kNone,
      cutlass::ComplexTransform::kNone,
      true,
      false,
      true>;
};
struct cutlass_tensorop_s1688gemm_64x64_16x3_nn_align4 {
  using Gemm = cutlass::gemm::device::GemmUniversal<
      float,
      cutlass::layout::RowMajor,
      float,
      cutlass::layout::RowMajor,
      float,
      cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 64, 16>,
      cutlass::gemm::GemmShape<32, 32, 16>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      3,
      4,
      4,
      cutlass::arch::OpMultiplyAddFastF32,
      cutlass::ComplexTransform::kNone,
      cutlass::ComplexTransform::kNone,
      true,
      false,
      true>;
};
struct cutlass_tensorop_d884gemm_16x32_16x5_nn_align1 {
  using Gemm = cutlass::gemm::device::GemmUniversal<
      double,
      cutlass::layout::RowMajor,
      double,
      cutlass::layout::RowMajor,
      double,
      cutlass::layout::RowMajor,
      double,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<16, 32, 16>,
      cutlass::gemm::GemmShape<16, 16, 16>,
      cutlass::gemm::GemmShape<8, 8, 4>,
      cutlass::epilogue::thread::LinearCombination<double, 1, double, double>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      5,
      1,
      1,
      cutlass::arch::OpMultiplyAdd,
      cutlass::ComplexTransform::kNone,
      cutlass::ComplexTransform::kNone,
      true,
      false,
      true>;
};
struct cutlass_tensorop_d884gemm_32x16_16x5_nn_align1 {
  using Gemm = cutlass::gemm::device::GemmUniversal<
      double,
      cutlass::layout::RowMajor,
      double,
      cutlass::layout::RowMajor,
      double,
      cutlass::layout::RowMajor,
      double,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<32, 16, 16>,
      cutlass::gemm::GemmShape<16, 16, 16>,
      cutlass::gemm::GemmShape<8, 8, 4>,
      cutlass::epilogue::thread::LinearCombination<double, 1, double, double>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      5,
      1,
      1,
      cutlass::arch::OpMultiplyAdd,
      cutlass::ComplexTransform::kNone,
      cutlass::ComplexTransform::kNone,
      true,
      false,
      true>;
};

// sm75
struct cutlass_tensorop_s1688gemm_f16_64x64_32x2_nn_align4 {
  using Gemm = cutlass::gemm::device::GemmUniversal<
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      float,
      cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<32, 32, 32>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      2,
      8,
      8,
      cutlass::arch::OpMultiplyAdd>;
};

}  // namespace sparse
}  // namespace phi
#endif
