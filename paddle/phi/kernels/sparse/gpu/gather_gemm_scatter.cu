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

#ifdef PADDLE_WITH_CUTLASS
#include "paddle/phi/kernels/sparse/gpu/gather_gemm_scatter.h"
namespace phi {
namespace sparse {
fp16_gather_gemm_scatter getBestFp16Kernel(const int M,
                                           const int N,
                                           const int K) {
  if (K == 4 && N == 16) {
    return launchKernel<cutlass::half_t,
                        cutlass_tensorop_h1688gemm_64x64_32x2_nn_align4::Gemm>;
  }
  if (K == 16 && N == 16) {
    return launchKernel<cutlass::half_t,
                        cutlass_tensorop_h1688gemm_64x64_32x2_nn_align8::Gemm>;
  }
  if (K == 16 && N == 32) {
    return launchKernel<cutlass::half_t,
                        cutlass_tensorop_h1688gemm_64x64_32x2_nn_align8::Gemm>;
  }
  if (K == 32 && N == 32) {
    return launchKernel<cutlass::half_t,
                        cutlass_tensorop_h1688gemm_64x64_32x2_nn_align8::Gemm>;
  }
  if (K == 32 && N == 64) {
    return launchKernel<cutlass::half_t,
                        cutlass_tensorop_h1688gemm_64x64_32x2_nn_align8::Gemm>;
  }
  if (K == 64 && N == 64) {
    if (M > 100000)
      launchKernel<
          cutlass::half_t,
          cutlass_tensorop_f16_s1688gemm_f16_64x128_32x2_nn_align8::Gemm>;
    if (M > 20000)
      launchKernel<
          cutlass::half_t,
          cutlass_tensorop_f16_s1688gemm_f16_64x64_32x2_nn_align8::Gemm>;
    if (M > 15000)
      return launchKernel<
          cutlass::half_t,
          cutlass_tensorop_h1688gemm_128x64_32x2_nn_align8::Gemm>;
    return launchKernel<cutlass::half_t,
                        cutlass_tensorop_h1688gemm_64x64_32x2_nn_align8::Gemm>;
  }
  if (K == 128) {
    if (M >= 5000)
      return launchKernel<
          cutlass::half_t,
          cutlass_tensorop_h1688gemm_64x64_32x2_nn_align8::Gemm>;
    return launchKernel<cutlass::half_t,
                        cutlass_tensorop_h16816gemm_64x64_64x5_nn_align8::Gemm>;
  }
  if (N == 128) {
    return launchKernel<cutlass::half_t,
                        cutlass_tensorop_h1688gemm_64x64_32x2_nn_align8::Gemm>;
  }
  return launchKernel<cutlass::half_t,
                      cutlass_tensorop_h1688gemm_64x64_32x2_nn_align4::Gemm>;
}
fp32_gather_gemm_scatter getBestFp32Kernel(const int M,
                                           const int N,
                                           const int K,
                                           const int SM) {
  if (SM == 75) {
    return launchKernel<
        float,
        cutlass_tensorop_s1688gemm_f16_64x64_32x2_nn_align4::Gemm>;
  }
  if (K == 4 && N == 16) {
    return launchKernel<
        float,
        cutlass_tensorop_s1688f16gemm_64x64_16x10_nn_align4::Gemm>;
  }
  if (K == 16 && N == 16) {
    return launchKernel<
        float,
        cutlass_tensorop_s1688f16gemm_64x64_16x10_nn_align4::Gemm>;
  }
  if (K == 16 && N == 32) {
    if (M >= 10000)
      return launchKernel<
          float,
          cutlass_tensorop_s1688gemm_64x64_16x3_nn_align4::Gemm>;
    return launchKernel<
        float,
        cutlass_tensorop_s1688f16gemm_64x64_16x10_nn_align4::Gemm>;
  }
  if (K == 32 && N == 32) {
    if (M >= 10000)
      return launchKernel<
          float,
          cutlass_tensorop_s1688gemm_64x64_16x3_nn_align4::Gemm>;
    return launchKernel<
        float,
        cutlass_tensorop_s1688f16gemm_64x64_16x10_nn_align4::Gemm>;
  }
  if (K == 32 && N == 64) {
    if (M >= 10000)
      return launchKernel<
          float,
          cutlass_tensorop_s1688gemm_64x64_16x3_nn_align4::Gemm>;
    return launchKernel<
        float,
        cutlass_tensorop_s1688f16gemm_64x64_16x10_nn_align4::Gemm>;
  }
  if (K == 64 && N == 64) {
    if (M >= 15000)
      return launchKernel<
          float,
          cutlass_tensorop_s1688gemm_64x64_16x3_nn_align4::Gemm>;
    return launchKernel<
        float,
        cutlass_tensorop_s1688f16gemm_64x64_16x10_nn_align4::Gemm>;
  }
  if (K == 128) {
    if (M >= 100000)
      return launchKernel<
          float,
          cutlass_tensorop_s1688f16gemm_128x128_16x3_nn_align4::Gemm>;
    if (M >= 5000)
      return launchKernel<
          float,
          cutlass_tensorop_s1688f16gemm_256x64_16x4_nn_align4::Gemm>;
    return launchKernel<
        float,
        cutlass_tensorop_s1688tf32gemm_256x128_16x3_nn_align4::Gemm>;
  }
  if (N == 128) {
    if (M >= 100000)
      return launchKernel<
          float,
          cutlass_tensorop_s1688tf32gemm_256x128_16x3_nn_align4::Gemm>;
    if (M >= 5000)
      return launchKernel<
          float,
          cutlass_tensorop_s1688f16gemm_128x128_16x3_nn_align4::Gemm>;
    return launchKernel<
        float,
        cutlass_tensorop_s1688f16gemm_64x128_16x6_nn_align4::Gemm>;
  }
  return launchKernel<
      float,
      cutlass_tensorop_s1688f16gemm_64x64_16x10_nn_align4::Gemm>;
}
fp64_gather_gemm_scatter getBestFp64Kernel(const int M,
                                           const int N,
                                           const int K) {
  if (K == 4 && N == 16) {
    return launchKernel<double,
                        cutlass_tensorop_d884gemm_16x32_16x5_nn_align1::Gemm>;
  }
  if (K == 16 && N == 16) {
    if (M >= 10000)
      return launchKernel<double,
                          cutlass_tensorop_d884gemm_32x16_16x5_nn_align1::Gemm>;
    return launchKernel<double,
                        cutlass_tensorop_d884gemm_16x32_16x5_nn_align1::Gemm>;
  }
  if (K == 16 && N == 32) {
    return launchKernel<double,
                        cutlass_tensorop_d884gemm_32x16_16x5_nn_align1::Gemm>;
  }
  if (K == 32 && N == 32) {
    return launchKernel<double,
                        cutlass_tensorop_d884gemm_16x32_16x5_nn_align1::Gemm>;
  }
  if (K == 32 && N == 64) {
    return launchKernel<double,
                        cutlass_tensorop_d884gemm_32x16_16x5_nn_align1::Gemm>;
  }
  if (K == 64 && N == 64) {
    return launchKernel<double,
                        cutlass_tensorop_d884gemm_32x16_16x5_nn_align1::Gemm>;
  }
  return launchKernel<double,
                      cutlass_tensorop_d884gemm_32x16_16x5_nn_align1::Gemm>;
}

}  // namespace sparse
}  // namespace phi
#endif
