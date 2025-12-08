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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#else
#include <cuda_runtime_api.h>  // NOLINT
#include "cuda.h"              // NOLINT
#endif

#if (defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11060) || \
    defined(PADDLE_WITH_HIP)

#include "paddle/common/flags.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/scope_guard.h"
#include "paddle/utils/optional.h"
#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11060
#include "paddle/phi/backends/dynload/cublasLt.h"
#include "paddle/phi/backends/gpu/cuda/cuda_helper.h"
#include "paddle/phi/kernels/funcs/blas/blaslt_impl.cu.h"
#elif defined(PADDLE_WITH_HIP)
#include "paddle/phi/backends/dynload/hipblasLt.h"
#include "paddle/phi/backends/gpu/rocm/rocm_helper.h"
#include "paddle/phi/kernels/funcs/blas/blaslt_impl.hip.h"
#endif
namespace phi {

/*
#if (defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11000 && \
     !(defined(_WIN32) || defined(WIN32)))
      if (!FLAGS_use_legacy_gemm &&  // NOLINT
          x.place().GetType() == phi::AllocationType::GPU &&
          weight.place().GetType() == phi::AllocationType::GPU &&
          bias.place().GetType() == phi::AllocationType::GPU &&
          !bias.is_dist_tensor())  // NOLINT
          [[likely]] {             // NOLINT
        // TODO(Pan Zhaowu): Add proper broadcast logic for batchsize unaligned
        // batch-gemm. Currently handles: (B..., k) x (k, n) -> (B..., n), with
        // 1D or scalar bias.

        // --- Original input tensor dimensions and values ---
        const auto &x_original_shape = x.shape();
        const size_t x_ndim_original = x_original_shape.size();

        const auto &weight_original_shape = weight.shape();
        const size_t weight_ndim_original = weight_original_shape.size();

        const int64_t k_dim =
            x_original_shape[x_ndim_original - 1];  // Last dimension of X

        paddle::Tensor x_processed =
            x;  // Start with original, possibly reassign if reshaped
        paddle::Tensor weight_processed =
            weight;  // Start with original, possibly reassign if reshaped

        // If x is 1D (e.g., shape [k]), reshape it to [1, k] to fit the (B...,
        // k) x (k, n) pattern. This effectively treats a 1D vector as a row
        // vector for matrix multiplication.
        if (x_ndim_original == 1) [[unlikely]] {
          x_processed = reshape_ad_func(x, {1, k_dim});
        } else if (weight_ndim_original == 1) [[unlikely]] {  // NOLINT
          weight_processed = reshape_ad_func(weight, {k_dim, 1});
        }

        // --- Recalculate dimensions based on processed tensors ---
        const auto &x_shape_current = x_processed.shape();
        const size_t x_ndim_current = x_shape_current.size();

        const auto &weight_shape_current = weight_processed.shape();
        const size_t weight_ndim_current = weight_shape_current.size();

        const int64_t k_effective = x_shape_current[x_ndim_current - 1];
        const int64_t n_effective =
            weight_shape_current[weight_ndim_current - 1];

        // --- Determine the final output shape ---
        std::vector<int64_t> output_shape_vec = x_shape_current;
        output_shape_vec[x_ndim_current - 1] = n_effective;

        // If the original x was 1D, the processed x became [1, k].
        if (x_ndim_original == 1 && output_shape_vec.size() > 1 &&
            output_shape_vec[0] == 1) {
          output_shape_vec.erase(
              output_shape_vec
                  .begin());  // Remove the artificial batch dimension
        }

        // This is used for reshaping X into a 2D matrix for addmm_ad_func.
        const int64_t x_batch_numel =
            std::accumulate(output_shape_vec.begin(),
                            output_shape_vec.end() - 1,
                            1LL,
                            std::multiplies<int64_t>());

        // --- Bias handling and GEMM execution ---
        // The condition now uses the processed weight's shape.
        // This branch typically handles (B..., k) x (k, n) where n > 1.
        if (weight_shape_current[0] > 1 && weight_shape_current[1] > 1) {
          paddle::Tensor bias_1d =
              bias;  // Create a mutable copy if modification is needed
          // Align bias' shape to 'n_effective'. If bias.numel() != n_effective,
          // tile it.
          if (bias.numel() != n_effective) {
            bias_1d = tile_ad_func(bias, {static_cast<int64_t>(n_effective)});
          }
          // Execute fused GEMM with epilogue.
          auto [out, _] = fused_gemm_epilogue_ad_func(
              x_processed, weight_processed, bias_1d, false, false, "none");

          // If original x was 1D and output_shape_vec is 1D (i.e., [n]),
          // but fused_gemm_epilogue_ad_func returns a 2D tensor ([1, n]),
          // reshape it back to the desired 1D output shape.
          if (x_ndim_original == 1 && out.shape().size() == 2 &&
              output_shape_vec.size() == 1) {
            out = reshape_ad_func(out, output_shape_vec);
          }

          PyEval_RestoreThread(tstate);
          tstate = nullptr;
          return ToPyObject(out);
        } else {
          // This branch handles cases where weight_processed is effectively 2D
          // with one dimension being 1, e.g., (B..., k) x (k, 1) resulting in
          // (B..., 1). Or when weight_processed was originally 1D and reshaped
          // to [k, 1].

          // Reshape bias to [1, n_effective] then tile to [x_batch_numel, 1]
          // for addmm_ad_func.
          paddle::Tensor bias_2d = tile_ad_func(
              reshape_ad_func(bias, {1, n_effective}), {x_batch_numel, 1});

          // Perform matrix multiplication using addmm_ad_func.
          // x_processed is reshaped to 2D [x_batch_numel, k_effective] for the
          // multiplication.
          auto out = addmm_ad_func(
              bias_2d,
              reshape_ad_func(x_processed, {x_batch_numel, k_effective}),
              weight_processed,
              1.0,
              1.0);

          out = reshape_ad_func(out, output_shape_vec);

          PyEval_RestoreThread(tstate);
          tstate = nullptr;
          return ToPyObject(out);
        }
      } else  // NOLINT(readability/braces)
#endif
*/
// we don't receive 2+d tensor as weight
inline std::tuple<int64_t, int64_t, int64_t> canonicalize_dims(
    const DenseTensor& input, const DenseTensor& weight) {
  const auto x_dims = input.dims();
  const auto y_dims = weight.dims();
  PADDLE_ENFORCE_LE_GE(
      y_dims.size(),
      2,
      platform::errors::InvalidArgument("Y must be at most 2D"));

  const int64_t N = y_dims.size() < 2 ? 1 : y_dims[y_dims.size() - 1];
  const int64_t K = y_dims.size() < 2 ? y_dims[0] : y_dims[y_dims.size() - 2];

  int64_t M = x_dims.size() >= 2 ? x_dims[x_dims.size() - 2] : 1;
  if (x_dims.size() > 2) {
    // Accumulate the batch dims for input
    for (int64_t i = 0; i < x_dims.size() - 2; ++i) {
      M *= x_dims[i];
    }
  }

  return {M, N, K};
}
template <typename T, typename Context>
void LinearV2Kernel(const Context& dev_ctx,
                    const DenseTensor& input,
                    const DenseTensor& weight,
                    const DenseTensor& bias,
                    DenseTensor* out) {
  PADDLE_ENFORCE_LE(bias.dims().size(),
                    1,
                    platform::errors::InvalidArgument("Bias must be 1D"));
  if (out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }
#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION < 11060 || \
    defined(PADDLE_WITH_HIP)
  // NOTE(Pan Zhaowu): Fallback logic for legacy CUDA version or DCU.
  // TODO(Pan Zhaowu): Implement this
  return;
#else
  dev_ctx.template Alloc<T>(out, out->numel() * sizeof(T));
  const auto [M, N, K] = canonicalize_dims(input, weight);

  if (bias.numel() != N) {
    // only broadcast to 1D bias whatsoever
    // pass1: scalar to 1D
  }
  if (N > 1 && K > 1) {
    // CublasLt path with bias add epilogue
    phi::funcs::LinearWithCublasLt<T>::Run(
        dev_ctx,
        &input,
        &weight,
        out,
        static_cast<const void*>(bias.data<T>()),
        nullptr,
        M,
        N,
        K,
        false,
        false,
        phi::funcs::MatmulFusedType::kMatmulBias);
  } else {
    // Cublas path with beta==1 bias adding.
    blas.GEMM(dev_ctx, )
  }
#endif
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
