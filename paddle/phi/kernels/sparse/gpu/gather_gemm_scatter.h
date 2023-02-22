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
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/kernels/sparse/gpu/common.h"
#include "paddle/phi/kernels/sparse/gpu/cutlass/build/generated/gemm/all_gemm_operations.h"
#include "paddle/phi/kernels/sparse/gpu/cutlass_tuner.h"

namespace phi {
namespace sparse {

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

template <typename T>
void GatherGemmScatter(const phi::GPUContext& ctx,
                       const T* const a,
                       const T* const b,
                       const T* const c,
                       T* const d,
                       const int& m,
                       const int& n,
                       const int& k,
                       const int32_t* a_indices,
                       const int32_t* b_indices,
                       const int32_t* c_d_indices,
                       T alpha,
                       T beta) {
  auto* tuner = MakeCutlassTuner<T>(fp16_kernels[0]);
  for (auto i = 1; i < fp16_kernels.size(); i++)
    tuner->AddCallBack(fp16_kernels[i]);

  size_t key = autotune::GenKey(m, n, k);

  tuner->CutlassRun(ctx,
                    key,
                    a,
                    b,
                    c,
                    d,
                    m,
                    n,
                    k,
                    a_indices,
                    b_indices,
                    c_d_indices,
                    alpha,
                    beta);
}

static void dispatchKernel(const GPUContext& dev_ctx,
                           const void* const a,
                           const void* const b,
                           const void* const c,
                           void* const d,
                           const int m,
                           const int n,
                           const int k,
                           const void* a_indices,
                           const void* c_d_indices,
                           const bool cutlass,
                           const phi::DataType type) {
  if (!cutlass) return;

  if (type == phi::DataType::FLOAT16) {
#if 0
    fp16_gather_gemm_scatter gather_gemm_scatter = getBestFp16Kernel(m, n, k);
#endif
    GatherGemmScatter(dev_ctx,
                      static_cast<const phi::dtype::float16*>(a),
                      static_cast<const phi::dtype::float16*>(b),
                      static_cast<const phi::dtype::float16*>(c),
                      static_cast<phi::dtype::float16*>(d),
                      m,
                      n,
                      k,
                      static_cast<const int32_t*>(a_indices),
                      nullptr,
                      static_cast<const int32_t*>(c_d_indices),
                      static_cast<phi::dtype::float16>(1),
                      static_cast<phi::dtype::float16>(1));
  } else if (type == phi::DataType::FLOAT32) {
    fp32_gather_gemm_scatter gather_gemm_scatter =
        getBestFp32Kernel(m, n, k, dev_ctx.GetComputeCapability());
    gather_gemm_scatter(dev_ctx,
                        static_cast<const float*>(a),
                        static_cast<const float*>(b),
                        static_cast<const float*>(c),
                        static_cast<float*>(d),
                        m,
                        n,
                        k,
                        static_cast<const int32_t*>(a_indices),
                        static_cast<const int32_t*>(c_d_indices),
                        static_cast<float>(1),
                        static_cast<float>(1));
  } else if (type == phi::DataType::FLOAT64) {
    fp64_gather_gemm_scatter gather_gemm_scatter = getBestFp64Kernel(m, n, k);
    gather_gemm_scatter(dev_ctx,
                        static_cast<const double*>(a),
                        static_cast<const double*>(b),
                        static_cast<const double*>(c),
                        static_cast<double*>(d),
                        m,
                        n,
                        k,
                        static_cast<const int32_t*>(a_indices),
                        static_cast<const int32_t*>(c_d_indices),
                        static_cast<double>(1),
                        static_cast<double>(1));
  }
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
