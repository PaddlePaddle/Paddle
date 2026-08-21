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
void LaunchAttentionLayerNormBackward(
    const GPUContext& dev_ctx,
    float epsilon,
    int rows,
    int cols,
    const T* x,
    const T* out_grad,
    const FusedLayerNormParamType<T>* scale,
    const FusedLayerNormParamType<T>* mean,
    const FusedLayerNormParamType<T>* variance,
    T* x_grad,
    FusedLayerNormParamType<T>* scale_grad,
    FusedLayerNormParamType<T>* bias_grad) {
  AttnLayerNorm<T> layer_norm(dev_ctx, epsilon, rows, cols);
  layer_norm.ComputeBackward(
      x, out_grad, scale, mean, variance, x_grad, scale_grad, bias_grad);
}

template <typename T>
void LaunchAttentionMatMulBackward(const GPUContext& dev_ctx,
                                   bool trans_a,
                                   bool trans_b,
                                   int m,
                                   int n,
                                   int k,
                                   bool compute_bias,
                                   const DenseTensor* input,
                                   const DenseTensor* weight,
                                   const DenseTensor* output_grad,
                                   DenseTensor* input_grad,
                                   DenseTensor* weight_grad,
                                   DenseTensor* bias_grad,
                                   bool use_addto,
                                   bool fused) {
  AttnMatMul<T> matmul(dev_ctx, trans_a, trans_b, m, n, k, compute_bias);
  matmul.ComputeBackward(input,
                         weight,
                         output_grad,
                         input_grad,
                         weight_grad,
                         bias_grad,
                         use_addto,
                         fused);
}

template <typename T>
void LaunchFMHABackward(const GPUContext& dev_ctx,
                        int batch_size,
                        int sequence_length,
                        int num_heads,
                        int head_dim,
                        const AttentionDropoutConfig& dropout,
                        const DenseTensor& transpose_out,
                        const DenseTensor* src_mask,
                        const DenseTensor& softmax_out,
                        const DenseTensor& dropout_mask_out,
                        const DenseTensor& dropout_out,
                        const DenseTensor& qk_out,
                        const DenseTensor& src_mask_out,
                        const DenseTensor& fmha_out_grad,
                        DenseTensor* qktv_out_grad,
                        DenseTensor* dropout_out_grad,
                        DenseTensor* softmax_out_grad,
                        DenseTensor* src_mask_out_grad,
                        DenseTensor* qk_out_grad,
                        DenseTensor* transpose_out_grad,
                        DenseTensor* cache_kv_grad,
                        DenseTensor* qkv_grad) {
  auto dropout_param = MakeAttentionDropoutParam(dropout);
  FMHARef<T> fmha(
      dev_ctx, batch_size, sequence_length, num_heads, head_dim, dropout_param);
  fmha.ComputeBackward(transpose_out,
                       src_mask,
                       softmax_out,
                       dropout_mask_out,
                       dropout_out,
                       qk_out,
                       src_mask_out,
                       fmha_out_grad,
                       qktv_out_grad,
                       dropout_out_grad,
                       softmax_out_grad,
                       src_mask_out_grad,
                       qk_out_grad,
                       transpose_out_grad,
                       cache_kv_grad,
                       qkv_grad);
}

#define INSTANTIATE_FUSED_ATTENTION_BACKWARD(T, U)                     \
  template void LaunchAttentionLayerNormBackward<T>(const GPUContext&, \
                                                    float,             \
                                                    int,               \
                                                    int,               \
                                                    const T*,          \
                                                    const T*,          \
                                                    const U*,          \
                                                    const U*,          \
                                                    const U*,          \
                                                    T*,                \
                                                    U*,                \
                                                    U*);               \
  template void LaunchAttentionMatMulBackward<T>(const GPUContext&,    \
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
                                                 DenseTensor*,         \
                                                 bool,                 \
                                                 bool);                \
  template void LaunchFMHABackward<T>(const GPUContext&,               \
                                      int,                             \
                                      int,                             \
                                      int,                             \
                                      int,                             \
                                      const AttentionDropoutConfig&,   \
                                      const DenseTensor&,              \
                                      const DenseTensor*,              \
                                      const DenseTensor&,              \
                                      const DenseTensor&,              \
                                      const DenseTensor&,              \
                                      const DenseTensor&,              \
                                      const DenseTensor&,              \
                                      const DenseTensor&,              \
                                      DenseTensor*,                    \
                                      DenseTensor*,                    \
                                      DenseTensor*,                    \
                                      DenseTensor*,                    \
                                      DenseTensor*,                    \
                                      DenseTensor*,                    \
                                      DenseTensor*,                    \
                                      DenseTensor*)

INSTANTIATE_FUSED_ATTENTION_BACKWARD(float, float);
INSTANTIATE_FUSED_ATTENTION_BACKWARD(double, double);
INSTANTIATE_FUSED_ATTENTION_BACKWARD(phi::float16, float);

#undef INSTANTIATE_FUSED_ATTENTION_BACKWARD

}  // namespace phi::fusion
