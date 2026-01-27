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
#include <algorithm>
#include <mutex>
#include <unordered_map>
#include "glog/logging.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

#include "paddle/common/enforce.h"
#include "paddle/phi/kernels/addmm_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/impl/matmul_kernel_impl.h"
#include "paddle/phi/kernels/linear_v2_kernel.h"
#include "paddle/phi/kernels/reshape_kernel.h"
#include "paddle/phi/kernels/tile_kernel.h"

#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#else
#include <cuda_runtime_api.h>  // NOLINT
#include "cuda.h"              // NOLINT
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#include "paddle/common/flags.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/scope_guard.h"
#include "paddle/utils/optional.h"
#if defined(PADDLE_WITH_CUDA)
#include "paddle/phi/backends/dynload/cublasLt.h"
#include "paddle/phi/backends/gpu/cuda/cuda_helper.h"
#include "paddle/phi/kernels/funcs/blas/blaslt_impl.cu.h"
#elif defined(PADDLE_WITH_HIP)
#include "paddle/phi/backends/dynload/hipblasLt.h"
#include "paddle/phi/backends/gpu/rocm/rocm_helper.h"
#include "paddle/phi/kernels/funcs/blas/blaslt_impl.hip.h"
#endif

#endif
COMMON_DECLARE_bool(use_legacy_linear);

namespace phi {

template <typename T, typename Context>
void LinearV2Kernel(const Context& dev_ctx,
                    const DenseTensor& input,
                    const DenseTensor& weight,
                    const DenseTensor& bias,
                    DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }

// broadcast bias, reshape input,  run_fuse, reshape output
#if defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP) && !defined(_WIN32)
  if (!FLAGS_use_legacy_linear) {
    VLOG(10) << "Use LinearV2Kernel with cublaslt";
    const auto out_dim_original = out->dims();
    const auto [M, N, K] = canonicalize_dims(input, weight);
    VLOG(10) << "M: " << M << ", N: " << N << ", K: " << K;
    DenseTensor input_processed = input;
    DenseTensor weight_processed = weight;
    input_processed.Resize(common::make_ddim({M, K}));
    weight_processed.Resize(common::make_ddim({K, N}));
    out->Resize(common::make_ddim({M, N}));
    VLOG(10) << "input_processed: " << input_processed.dims()
             << ", weight_processed: " << weight_processed.dims()
             << ", output_processed: " << out->dims();

    if (N > 1 && K > 1) {
      DenseTensor bias_processed;
      if (bias.numel() != N) {
        // only broadcast to 1D bias whatsoever
        // pass1: scalar to 1D
        phi::TileKernel<T, Context>(dev_ctx, bias, {N}, &bias_processed);
      } else {
        bias_processed = bias;
      }
      // CublasLt path with bias add epilogue
      phi::funcs::LinearWithCublasLt<T>::Run(
          dev_ctx,
          &input_processed,
          &weight_processed,
          out,
          static_cast<const void*>(bias_processed.data<T>()),
          nullptr,
          M,
          N,
          K,
          false,
          false,
          phi::funcs::MatmulFusedType::kMatmulBias);
    } else {
      DenseTensor bias_processed = bias;
      if (bias.numel() != (M * N)) {
        bias_processed.Resize(common::make_ddim({1, bias.numel()}));
        VLOG(10) << "bias.dim(): " << bias.dims();
        VLOG(10) << "M*N: " << M * N;
        VLOG(10) << "bias tiling and addmm calculating";
        // only broadcast to 1D bias whatsoever
        phi::TileKernel<T, Context>(
            dev_ctx, bias_processed, {M, 1}, &bias_processed);
        VLOG(10) << "bias_processed.dims(): " << bias_processed.dims();
      } else {
        bias_processed = bias;
      }
      phi::AddmmKernel<T>(dev_ctx,
                          bias_processed,
                          input_processed,
                          weight_processed,
                          1.0f,
                          1.0f,
                          out);
    }
    VLOG(10) << "linear calculate complete";
    out->Resize(out_dim_original);
  } else  // NOLINT
#endif
  // Fallback logic for legacy CUDA version or other hardware.
  // Or specified by user to use a legacy behaviour.
  {  // NOLINT
    // NOTE(Pan Zhaowu): Fallback logic for legacy CUDA version or DCU.
    std::vector<std::int64_t> input_dims_vec = common::vectorize(input.dims());
    std::vector<std::int64_t> weight_dims_vec =
        common::vectorize(weight.dims());

    MatMulFunction<Context, T>(dev_ctx,
                               input,
                               weight,
                               input_dims_vec,
                               weight_dims_vec,
                               out,
                               false,
                               false);
    AddKernel<T, Context>(dev_ctx, *out, bias, out);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(linear_v2,
                   GPU,
                   ALL_LAYOUT,
                   phi::LinearV2Kernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {}
