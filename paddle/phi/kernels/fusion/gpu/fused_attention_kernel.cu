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

#include "paddle/phi/kernels/fusion/gpu/fused_transformer_kernel_launch.h"

#include "paddle/phi/kernels/fusion/gpu/attention_layer.norm.h"
#include "paddle/phi/kernels/fusion/gpu/attn_gemm.h"
#include "paddle/phi/kernels/fusion/gpu/fmha_ref.h"

namespace phi::fusion {

namespace {

AttnDropoutParam MakeAttentionDropoutParam(
    const AttentionDropoutConfig& config) {
  return AttnDropoutParam(config.is_test,
                          *config.implementation,
                          config.probability,
                          config.is_upscale_in_train,
                          config.fix_seed,
                          config.seed_value,
                          config.seed);
}

}  // namespace

template <typename T>
void LaunchAttentionLayerNormForward(const GPUContext& dev_ctx,
                                     float epsilon,
                                     int rows,
                                     int cols,
                                     const T* x,
                                     const FusedLayerNormParamType<T>* scale,
                                     const FusedLayerNormParamType<T>* bias,
                                     T* out,
                                     FusedLayerNormParamType<T>* mean,
                                     FusedLayerNormParamType<T>* variance) {
  AttnLayerNorm<T> layer_norm(dev_ctx, epsilon, rows, cols);
  layer_norm.ComputeForward(x, scale, bias, out, mean, variance);
}

template <typename T>
void LaunchAttentionMatMulForward(const GPUContext& dev_ctx,
                                  bool trans_a,
                                  bool trans_b,
                                  int m,
                                  int n,
                                  int k,
                                  bool compute_bias,
                                  const DenseTensor* weight,
                                  const DenseTensor* input,
                                  const DenseTensor* bias,
                                  DenseTensor* output,
                                  DenseTensor* bias_out,
                                  bool fused) {
  AttnMatMul<T> matmul(dev_ctx, trans_a, trans_b, m, n, k, compute_bias);
  matmul.ComputeForward(weight, input, bias, output, bias_out, fused);
}

template <typename T>
void LaunchFMHAForward(const GPUContext& dev_ctx,
                       int batch_size,
                       int sequence_length,
                       int num_heads,
                       int head_dim,
                       const AttentionDropoutConfig& dropout,
                       const DenseTensor& qkv,
                       const DenseTensor* cache_kv,
                       const DenseTensor* src_mask,
                       DenseTensor* transpose_out,
                       DenseTensor* cache_kv_out,
                       DenseTensor* qk_out,
                       DenseTensor* src_mask_out,
                       DenseTensor* softmax_out,
                       DenseTensor* dropout_mask_out,
                       DenseTensor* dropout_out,
                       DenseTensor* qktv_out,
                       DenseTensor* fmha_out) {
  auto dropout_param = MakeAttentionDropoutParam(dropout);
  FMHARef<T> fmha(
      dev_ctx, batch_size, sequence_length, num_heads, head_dim, dropout_param);
  fmha.ComputeForward(qkv,
                      cache_kv,
                      src_mask,
                      transpose_out,
                      cache_kv_out,
                      qk_out,
                      src_mask_out,
                      softmax_out,
                      dropout_mask_out,
                      dropout_out,
                      qktv_out,
                      fmha_out);
}

#define INSTANTIATE_FUSED_ATTENTION_FORWARD(T, U)                     \
  template void LaunchAttentionLayerNormForward<T>(const GPUContext&, \
                                                   float,             \
                                                   int,               \
                                                   int,               \
                                                   const T*,          \
                                                   const U*,          \
                                                   const U*,          \
                                                   T*,                \
                                                   U*,                \
                                                   U*);               \
  template void LaunchAttentionMatMulForward<T>(const GPUContext&,    \
                                                bool,                 \
                                                bool,                 \
                                                int,                  \
                                                int,                  \
                                                int,                  \
                                                bool,                 \
                                                const DenseTensor*,   \
                                                const DenseTensor*,   \
                                                const DenseTensor*,   \
                                                DenseTensor*,         \
                                                DenseTensor*,         \
                                                bool);                \
  template void LaunchFMHAForward<T>(const GPUContext&,               \
                                     int,                             \
                                     int,                             \
                                     int,                             \
                                     int,                             \
                                     const AttentionDropoutConfig&,   \
                                     const DenseTensor&,              \
                                     const DenseTensor*,              \
                                     const DenseTensor*,              \
                                     DenseTensor*,                    \
                                     DenseTensor*,                    \
                                     DenseTensor*,                    \
                                     DenseTensor*,                    \
                                     DenseTensor*,                    \
                                     DenseTensor*,                    \
                                     DenseTensor*,                    \
                                     DenseTensor*,                    \
                                     DenseTensor*)

INSTANTIATE_FUSED_ATTENTION_FORWARD(float, float);
INSTANTIATE_FUSED_ATTENTION_FORWARD(double, double);
INSTANTIATE_FUSED_ATTENTION_FORWARD(phi::float16, float);

#undef INSTANTIATE_FUSED_ATTENTION_FORWARD

}  // namespace phi::fusion
