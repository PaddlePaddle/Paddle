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

#pragma once
#ifdef PADDLE_WITH_CUTLASS
#include "paddle/phi/kernels/sparse/gpu/cutlass_generator/common.h"
#include "paddle/phi/kernels/sparse/gpu/cutlass_generator/configurations.h"

namespace phi {
namespace sparse {
static std::vector<gather_hgemm_scatter> sm75_fp16_nn_kernels = {
    launchKernel<cutlass_tensorop_h1688gemm_256x128_32x2_nn_align8<>>,
};
static std::vector<gather_sgemm_f16_scatter> sm75_fp32_nn_kernels = {
    launchKernel<cutlass_tensorop_s1688gemm_f16_256x128_32x2_nn_align8<>>,
};
static std::vector<gather_hgemm_scatter> sm80_fp16_nn_kernels = {
    launchKernel<cutlass_tensorop_h16816gemm_256x128_32x3_nn_align8<>>,
};
static std::vector<gather_sgemm_scatter> sm80_fp32_nn_kernels = {
    launchKernel<cutlass_tensorop_s1688gemm_tf32_256x128_16x3_nn_align4<>>,
    launchKernel<cutlass_tensorop_tf32_s1688gemm_tf32_256x128_16x3_nn_align4<>>,
    launchKernel<cutlass_tensorop_s1688tf32gemm_256x128_16x3_nn_align4<>>,
    launchKernel<cutlass_tensorop_s1688gemm_128x128_16x4_nn_align4<>>,
};
static std::vector<gather_sgemm_scatter> sm80_fp32_nt_kernels = {
    launchKernel<cutlass_tensorop_s1688gemm_tf32_256x128_16x3_nt_align4<>>,
    launchKernel<cutlass_tensorop_tf32_s1688gemm_tf32_256x128_16x3_nt_align4<>>,
    launchKernel<cutlass_tensorop_s1688tf32gemm_256x128_16x3_nt_align4<>>,
    launchKernel<cutlass_tensorop_s1688gemm_128x128_16x4_nt_align4<>>,
};
static std::vector<gather_sgemm_scatter> sm80_fp32_tn_kernels = {
    launchKernel<cutlass_tensorop_s1688gemm_tf32_256x128_16x3_tn_align4<
        cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel>>,
    launchKernel<cutlass_tensorop_tf32_s1688gemm_tf32_256x128_16x3_tn_align4<
        cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel>>,
    launchKernel<cutlass_tensorop_s1688tf32gemm_256x128_16x3_tn_align4<
        cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel>>,
    launchKernel<cutlass_tensorop_s1688gemm_128x128_16x4_tn_align4<
        cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel>>,
};

}  // namespace sparse
}  // namespace phi
#endif
