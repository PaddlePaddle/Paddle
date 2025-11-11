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

/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. */

/*This code is copied from NVIDIA apex:
 *     https://github.com/NVIDIA/apex
 *     with minor changes. */

#include "ln.h"                // NOLINT
#include "ln_bwd_kernels.h"    // NOLINT
#include "ln_kernel_traits.h"  // NOLINT
#include "ln_utils.h"          // NOLINT

using namespace layer_norm;  // NOLINT

template <typename weight_t,
          typename input_t,
          typename output_t,
          typename compute_t,
          typename index_t,
          int HIDDEN_SIZE,
          int CTAS_PER_ROW,
          int WARPS_M,
          int WARPS_N,
          int BYTES_PER_LDG_MAIN,
          int BYTES_PER_LDG_FINAL>
void launch_(LaunchParams<BwdParams> &launch_params,  // NOLINT
             const bool configure_params) {
  using KernelTraits = KernelTraits<weight_t,
                                    input_t,
                                    output_t,
                                    compute_t,
                                    index_t,
                                    HIDDEN_SIZE,
                                    CTAS_PER_ROW,
                                    WARPS_M,
                                    WARPS_N,
                                    BYTES_PER_LDG_MAIN>;
  auto kernel = &ln_bwd_kernel<KernelTraits>;

  if (configure_params) {
    int ctas_per_sm;
    cudaError status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &ctas_per_sm,
        kernel,
        KernelTraits::THREADS_PER_CTA,
        KernelTraits::SMEM_BYTES);
    launch_params.params.ctas_per_col =
        launch_params.props->multiProcessorCount * ctas_per_sm /
        KernelTraits::CTAS_PER_ROW;
    launch_params.barrier_size = 0;
    launch_params.workspace_bytes = 0;
    if (KernelTraits::CTAS_PER_ROW > 1) {
      launch_params.barrier_size = 2 * launch_params.params.ctas_per_col;
      launch_params.workspace_bytes =
          launch_params.params.ctas_per_col * KernelTraits::WARPS_M *
          KernelTraits::CTAS_PER_ROW * sizeof(typename KernelTraits::reduce_t) *
          2;
    }
    return;
  }

  if (KernelTraits::SMEM_BYTES >= 48 * 1024) {
    CHECK_CUDA(cudaFuncSetAttribute(kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    KernelTraits::SMEM_BYTES));
  }
  auto stream = launch_params.stream;
  auto ctas_per_col = launch_params.params.ctas_per_col;

  if (KernelTraits::CTAS_PER_ROW == 1) {
    kernel<<<ctas_per_col,
             KernelTraits::THREADS_PER_CTA,
             KernelTraits::SMEM_BYTES,
             stream>>>(launch_params.params);
  } else {
    dim3 grid(KernelTraits::CTAS_PER_ROW * ctas_per_col);
    dim3 block(KernelTraits::THREADS_PER_CTA);
    void *params_ = (void *)&launch_params.params;  // NOLINT
    cudaLaunchCooperativeKernel((void *)kernel,     // NOLINT
                                grid,
                                block,
                                (void **)&params_,  // NOLINT
                                KernelTraits::SMEM_BYTES,
                                stream);
  }

  using KernelTraitsF =
      layer_norm::KernelTraitsFinalize<HIDDEN_SIZE,
                                       weight_t,
                                       input_t,
                                       output_t,
                                       compute_t,
                                       index_t,
                                       32 * 32,  // THREADS_PER_CTA
                                       BYTES_PER_LDG_FINAL>;

  auto kernel_f = &layer_norm::ln_bwd_finalize_kernel<KernelTraitsF>;
  kernel_f<<<KernelTraitsF::CTAS, KernelTraitsF::THREADS_PER_CTA, 0, stream>>>(
      launch_params.params);
}

// Create backward launch function and register. Macro signature:
//  HIDDEN_SIZE, WTYPE, ITYPE, OTYPE, CTYPE, CTAS_PER_ROW, WARPS_M, WARPS_N,
//  BYTES_PER_LDG, BYTES_PER_LDG_FINAL

REGISTER_BWD_LAUNCHER(768, fp32, fp32, fp32, fp32, 1, 4, 1, 16, 4);
REGISTER_BWD_LAUNCHER(768, fp16, fp16, fp16, fp32, 1, 4, 1, 16, 4);
REGISTER_BWD_LAUNCHER(768, fp16, fp32, fp16, fp32, 1, 4, 1, 16, 4);
REGISTER_BWD_LAUNCHER(768, bf16, bf16, bf16, fp32, 1, 4, 1, 16, 4);
REGISTER_BWD_LAUNCHER(768, bf16, fp32, bf16, fp32, 1, 4, 1, 16, 4);

REGISTER_BWD_LAUNCHER(1024, fp32, fp32, fp32, fp32, 1, 4, 1, 16, 4);
REGISTER_BWD_LAUNCHER(1024, fp16, fp16, fp16, fp32, 1, 4, 1, 16, 4);
REGISTER_BWD_LAUNCHER(1024, fp16, fp32, fp16, fp32, 1, 4, 1, 16, 4);
REGISTER_BWD_LAUNCHER(1024, bf16, bf16, bf16, fp32, 1, 4, 1, 16, 4);
REGISTER_BWD_LAUNCHER(1024, bf16, fp32, bf16, fp32, 1, 4, 1, 16, 4);

REGISTER_BWD_LAUNCHER(1536, fp32, fp32, fp32, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(1536, fp16, fp16, fp16, fp32, 1, 1, 4, 8, 4);
REGISTER_BWD_LAUNCHER(1536, fp16, fp32, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(1536, bf16, bf16, bf16, fp32, 1, 1, 4, 8, 4);
REGISTER_BWD_LAUNCHER(1536, bf16, fp32, bf16, fp32, 1, 1, 4, 16, 4);

REGISTER_BWD_LAUNCHER(2048, fp32, fp32, fp32, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(2048, fp16, fp16, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(2048, fp16, fp32, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(2048, bf16, bf16, bf16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(2048, bf16, fp32, bf16, fp32, 1, 1, 4, 16, 4);

REGISTER_BWD_LAUNCHER(2304, fp32, fp32, fp32, fp32, 1, 1, 4, 8, 4);
REGISTER_BWD_LAUNCHER(2304, fp16, fp16, fp16, fp32, 1, 1, 4, 4, 4);
REGISTER_BWD_LAUNCHER(2304, fp16, fp32, fp16, fp32, 1, 1, 4, 8, 4);
REGISTER_BWD_LAUNCHER(2304, bf16, bf16, bf16, fp32, 1, 1, 4, 4, 4);
REGISTER_BWD_LAUNCHER(2304, bf16, fp32, bf16, fp32, 1, 1, 4, 8, 4);

REGISTER_BWD_LAUNCHER(3072, fp32, fp32, fp32, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(3072, fp16, fp16, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(3072, fp16, fp32, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(3072, bf16, bf16, bf16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(3072, bf16, fp32, bf16, fp32, 1, 1, 4, 16, 4);

REGISTER_BWD_LAUNCHER(3840, fp32, fp32, fp32, fp32, 1, 1, 4, 8, 4);
REGISTER_BWD_LAUNCHER(3840, fp16, fp16, fp16, fp32, 1, 1, 4, 4, 4);
REGISTER_BWD_LAUNCHER(3840, fp16, fp32, fp16, fp32, 1, 1, 4, 8, 4);
REGISTER_BWD_LAUNCHER(3840, bf16, bf16, bf16, fp32, 1, 1, 4, 4, 4);
REGISTER_BWD_LAUNCHER(3840, bf16, fp32, bf16, fp32, 1, 1, 4, 8, 4);

REGISTER_BWD_LAUNCHER(4096, fp32, fp32, fp32, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(4096, fp16, fp16, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(4096, fp16, fp32, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(4096, bf16, bf16, bf16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(4096, bf16, fp32, bf16, fp32, 1, 1, 4, 16, 4);

REGISTER_BWD_LAUNCHER(5120, fp32, fp32, fp32, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(5120, fp16, fp16, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(5120, fp16, fp32, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(5120, bf16, bf16, bf16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(5120, bf16, fp32, bf16, fp32, 1, 1, 4, 16, 4);

REGISTER_BWD_LAUNCHER(6144, fp32, fp32, fp32, fp32, 1, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(6144, fp16, fp16, fp16, fp32, 1, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(6144, fp16, fp32, fp16, fp32, 1, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(6144, bf16, bf16, bf16, fp32, 1, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(6144, bf16, fp32, bf16, fp32, 1, 1, 8, 16, 4);

REGISTER_BWD_LAUNCHER(8192, fp32, fp32, fp32, fp32, 2, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(8192, fp16, fp16, fp16, fp32, 2, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(8192, fp16, fp32, fp16, fp32, 2, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(8192, bf16, bf16, bf16, fp32, 2, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(8192, bf16, fp32, bf16, fp32, 2, 1, 4, 16, 4);

REGISTER_BWD_LAUNCHER(10240, fp32, fp32, fp32, fp32, 2, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(10240, fp16, fp16, fp16, fp32, 2, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(10240, fp16, fp32, fp16, fp32, 2, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(10240, bf16, bf16, bf16, fp32, 2, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(10240, bf16, fp32, bf16, fp32, 2, 1, 4, 16, 4);

REGISTER_BWD_LAUNCHER(12288, fp32, fp32, fp32, fp32, 4, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(12288, fp16, fp16, fp16, fp32, 4, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(12288, fp16, fp32, fp16, fp32, 4, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(12288, bf16, bf16, bf16, fp32, 4, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(12288, bf16, fp32, bf16, fp32, 4, 1, 4, 16, 4);

REGISTER_BWD_LAUNCHER(12800, fp32, fp32, fp32, fp32, 5, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(12800, fp16, fp16, fp16, fp32, 5, 1, 4, 8, 4);
REGISTER_BWD_LAUNCHER(12800, fp16, fp32, fp16, fp32, 5, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(12800, bf16, bf16, bf16, fp32, 5, 1, 4, 8, 4);
REGISTER_BWD_LAUNCHER(12800, bf16, fp32, bf16, fp32, 5, 1, 4, 16, 4);

REGISTER_BWD_LAUNCHER(14336, fp32, fp32, fp32, fp32, 4, 1, 4, 8, 4);
REGISTER_BWD_LAUNCHER(14336, fp16, fp16, fp16, fp32, 4, 1, 4, 8, 4);
REGISTER_BWD_LAUNCHER(14336, fp16, fp32, fp16, fp32, 4, 1, 4, 8, 4);
REGISTER_BWD_LAUNCHER(14336, bf16, bf16, bf16, fp32, 4, 1, 4, 8, 4);
REGISTER_BWD_LAUNCHER(14336, bf16, fp32, bf16, fp32, 4, 1, 4, 8, 4);

REGISTER_BWD_LAUNCHER(15360, fp32, fp32, fp32, fp32, 4, 1, 4, 8, 4);
REGISTER_BWD_LAUNCHER(15360, fp16, fp16, fp16, fp32, 4, 1, 4, 4, 4);
REGISTER_BWD_LAUNCHER(15360, fp16, fp32, fp16, fp32, 4, 1, 4, 8, 4);
REGISTER_BWD_LAUNCHER(15360, bf16, bf16, bf16, fp32, 4, 1, 4, 4, 4);
REGISTER_BWD_LAUNCHER(15360, bf16, fp32, bf16, fp32, 4, 1, 4, 8, 4);

REGISTER_BWD_LAUNCHER(16384, fp32, fp32, fp32, fp32, 4, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(16384, fp16, fp16, fp16, fp32, 4, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(16384, fp16, fp32, fp16, fp32, 4, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(16384, bf16, bf16, bf16, fp32, 4, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(16384, bf16, fp32, bf16, fp32, 4, 1, 4, 16, 4);

REGISTER_BWD_LAUNCHER(18432, fp32, fp32, fp32, fp32, 4, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(18432, fp16, fp16, fp16, fp32, 4, 1, 4, 8, 4);
REGISTER_BWD_LAUNCHER(18432, fp16, fp32, fp16, fp32, 4, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(18432, bf16, bf16, bf16, fp32, 4, 1, 4, 8, 4);
REGISTER_BWD_LAUNCHER(18432, bf16, fp32, bf16, fp32, 4, 1, 4, 16, 4);

REGISTER_BWD_LAUNCHER(20480, fp32, fp32, fp32, fp32, 4, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(20480, fp16, fp16, fp16, fp32, 4, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(20480, fp16, fp32, fp16, fp32, 4, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(20480, bf16, bf16, bf16, fp32, 4, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(20480, bf16, fp32, bf16, fp32, 4, 1, 4, 16, 4);

REGISTER_BWD_LAUNCHER(24576, fp32, fp32, fp32, fp32, 4, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(24576, fp16, fp16, fp16, fp32, 4, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(24576, fp16, fp32, fp16, fp32, 4, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(24576, bf16, bf16, bf16, fp32, 4, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(24576, bf16, fp32, bf16, fp32, 4, 1, 8, 16, 4);

REGISTER_BWD_LAUNCHER(25600, fp32, fp32, fp32, fp32, 5, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(25600, fp16, fp16, fp16, fp32, 5, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(25600, fp16, fp32, fp16, fp32, 5, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(25600, bf16, bf16, bf16, fp32, 5, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(25600, bf16, fp32, bf16, fp32, 5, 1, 4, 16, 4);

REGISTER_BWD_LAUNCHER(30720, fp32, fp32, fp32, fp32, 4, 1, 8, 8, 4);
REGISTER_BWD_LAUNCHER(30720, fp16, fp16, fp16, fp32, 4, 1, 8, 4, 4);
REGISTER_BWD_LAUNCHER(30720, fp16, fp32, fp16, fp32, 4, 1, 8, 8, 4);
REGISTER_BWD_LAUNCHER(30720, bf16, bf16, bf16, fp32, 4, 1, 8, 4, 4);
REGISTER_BWD_LAUNCHER(30720, bf16, fp32, bf16, fp32, 4, 1, 8, 8, 4);

REGISTER_BWD_LAUNCHER(32768, fp32, fp32, fp32, fp32, 4, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(32768, fp16, fp16, fp16, fp32, 4, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(32768, fp16, fp32, fp16, fp32, 4, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(32768, bf16, bf16, bf16, fp32, 4, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(32768, bf16, fp32, bf16, fp32, 4, 1, 8, 16, 4);

REGISTER_BWD_LAUNCHER(40960, fp32, fp32, fp32, fp32, 4, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(40960, fp16, fp16, fp16, fp32, 4, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(40960, fp16, fp32, fp16, fp32, 4, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(40960, bf16, bf16, bf16, fp32, 4, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(40960, bf16, fp32, bf16, fp32, 4, 1, 8, 16, 4);

REGISTER_BWD_LAUNCHER(49152, fp32, fp32, fp32, fp32, 8, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(49152, fp16, fp16, fp16, fp32, 8, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(49152, fp16, fp32, fp16, fp32, 8, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(49152, bf16, bf16, bf16, fp32, 8, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(49152, bf16, fp32, bf16, fp32, 8, 1, 8, 16, 4);

REGISTER_BWD_LAUNCHER(65536, fp32, fp32, fp32, fp32, 8, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(65536, fp16, fp16, fp16, fp32, 8, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(65536, fp16, fp32, fp16, fp32, 8, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(65536, bf16, bf16, bf16, fp32, 8, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(65536, bf16, fp32, bf16, fp32, 8, 1, 8, 16, 4);
