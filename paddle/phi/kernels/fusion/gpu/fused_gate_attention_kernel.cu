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

#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/kernels/funcs/fused_gate_attention.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/fusion/gpu/attn_gemm.h"
#include "paddle/phi/kernels/fusion/gpu/fused_gate_attention_kernel_launch.h"

namespace phi {
namespace fusion {

template <typename T>
struct SigmoidMultiplyFunctor {
  using MT = typename MPTypeTrait<T>::Type;
  MT one = static_cast<MT>(1.0f);

  // sigmoid(x) = 1 / (1 + exp(-x))
  // out = sigmoid(x) * y
  inline HOSTDEVICE T operator()(T x, T y) const {
    MT x_mp = static_cast<MT>(x);
    T sigmoid_out = static_cast<T>(one / (one + exp(-x_mp)));
    return sigmoid_out * y;
  }
};

template <typename T>
void LaunchGateAttentionMergedQKVMatmulForward(
    const GPUContext& dev_ctx,
    const funcs::GateAttentionConfig<T>& config,
    const DenseTensor* query,
    DenseTensor* qkv_out,
    const DenseTensor& qkv_weight_in) {
  // query: shape=[batch_size, seq_len_m, seq_len_r, qkv_dim]
  // qkv_weight: shape=[3, num_heads, head_dim, qkv_dim]
  // qkv_out: shape=[batch_size, seq_len_m, seq_len_r, 3, num_heads, head_dim]
  auto* qkv_weight = &qkv_weight_in;

  // qkv_out = GEMM(query, qkv_weight^T)
  int m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int n = 3 * config.num_heads * config.head_dim;
  int k = config.q_dim;
  auto qkv_compute =
      fusion::AttnMatMul<T>(dev_ctx, false, true, m, n, k, false);
  qkv_compute.ComputeForward(qkv_weight, query, nullptr, qkv_out, nullptr);
}

template <typename T>
void LaunchGateAttentionSeparatedQKVMatmulForward(
    const GPUContext& dev_ctx,
    const funcs::GateAttentionConfig<T>& config,
    const DenseTensor* query,
    const DenseTensor* key,
    DenseTensor* query_out,
    DenseTensor* key_out,
    DenseTensor* value_out,
    const DenseTensor& query_weight_in,
    const DenseTensor& key_weight_in,
    const DenseTensor& value_weight_in) {
  auto* query_weight = &query_weight_in;
  auto* key_weight = &key_weight_in;
  auto* value_weight = &value_weight_in;

  // query_out = GEMM(query, query_weight)
  // query: shape=[batch_size, seq_len_m, seq_len_r, q_dim]
  // query_weight: shape=[q_dim, num_heads, head_dim]
  // query_out: shape=[batch_size, seq_len_m, seq_len_r, num_heads, head_dim]
  int q_m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int q_n = config.num_heads * config.head_dim;
  int q_k = config.q_dim;
  auto q_compute =
      fusion::AttnMatMul<T>(dev_ctx, false, false, q_m, q_n, q_k, false);
  q_compute.ComputeForward(query_weight, query, nullptr, query_out, nullptr);

  // k_out = GEMM(key, key_weight)
  // key: shape=[batch_size, seq_len_m, m_size, kv_dim]
  // key_weight: shape=[kv_dim, num_heads, head_dim]
  // key_out: shape=[batch_size, seq_len_m, m_size, num_heads, head_dim]
  int kv_m = config.batch_size * config.seq_len_m * config.m_size;
  int kv_n = config.num_heads * config.head_dim;
  int kv_k = config.kv_dim;
  auto kv_compute =
      fusion::AttnMatMul<T>(dev_ctx, false, false, kv_m, kv_n, kv_k, false);
  kv_compute.ComputeForward(key_weight, key, nullptr, key_out, nullptr);

  // value_out = GEMM(value, value_weight)
  kv_compute.ComputeForward(value_weight, key, nullptr, value_out, nullptr);
}

template <typename T>
void LaunchGateAttentionGatingLinearForward(
    const GPUContext& dev_ctx,
    const funcs::GateAttentionConfig<T>& config,
    const DenseTensor* query,
    const DenseTensor* fmha_out,
    DenseTensor* gate_bias_out,
    bool use_fused_matmul_bias,
    const DenseTensor& gate_weight_in,
    const DenseTensor& gate_bias_in) {
  auto* gate_weight = &gate_weight_in;
  auto* gate_bias = &gate_bias_in;

  // The first gate_bias_out stores the result of the multiplication,
  // and the second gate_bias_out stores the result of the multiplication +
  // bias.
  //   gate_out = GEMM(query, gate_weight) + gate_bias
  int m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int n = config.num_heads * config.head_dim;
  int k = config.q_dim;
  auto gate_linear =
      fusion::AttnMatMul<T>(dev_ctx, false, false, m, n, k, true);
  gate_linear.ComputeForward(gate_weight,
                             query,
                             gate_bias,
                             gate_bias_out,
                             gate_bias_out,
                             use_fused_matmul_bias);

  // gate_out = sigmoid(gate_out) * fmha_out
  std::vector<const DenseTensor*> ins = {gate_bias_out, fmha_out};
  std::vector<DenseTensor*> outs = {gate_bias_out};
  funcs::ElementwiseKernel<T>(dev_ctx, ins, &outs, SigmoidMultiplyFunctor<T>());
}

template <typename T>
void LaunchGateAttentionOutputLinearForward(
    const GPUContext& dev_ctx,
    const funcs::GateAttentionConfig<T>& config,
    const DenseTensor* fmha_or_gate_out,
    DenseTensor* out,
    bool use_fused_matmul_bias,
    const DenseTensor& out_linear_weight_in,
    const DenseTensor& out_linear_bias_in) {
  const auto* out_linear_weight = &out_linear_weight_in;
  const auto* out_linear_bias = &out_linear_bias_in;

  // out = GEMM(fmha_or_gate_out, out_linear_weight) + out_linear_bias
  int m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int n = config.q_dim;
  int k = config.num_heads * config.head_dim;
  auto out_linear = fusion::AttnMatMul<T>(dev_ctx, false, false, m, n, k, true);
  out_linear.ComputeForward(out_linear_weight,
                            fmha_or_gate_out,
                            out_linear_bias,
                            out,
                            out,
                            use_fused_matmul_bias);
}

template <typename T>
void LaunchGateAttentionFMHAForward(const GPUContext& dev_ctx,
                                    const DenseTensor* nonbatched_bias,
                                    const DenseTensor* src_mask,
                                    DenseTensor* query_transpose_out,
                                    DenseTensor* key_transpose_out,
                                    DenseTensor* value_transpose_out,
                                    DenseTensor* qkv_transpose_out,
                                    DenseTensor* softmax_out,
                                    DenseTensor* softmax_lse,
                                    DenseTensor* fmha_out,
                                    DenseTensor* gate_out,
                                    funcs::GateAttentionConfig<T>* config) {
  if (config->CanUseFlashAttn()) {
    auto fmha_compute = funcs::FlashAttnWithGating<T>(dev_ctx, true);
    fmha_compute.ComputeForward(nonbatched_bias,
                                src_mask,
                                qkv_transpose_out,
                                softmax_lse,
                                fmha_out,
                                config);
    return;
  }

  auto fmha_compute = funcs::FMHAGateRef<T>(dev_ctx, config->merge_qkv);
  fmha_compute.ComputeForward(nonbatched_bias,
                              src_mask,
                              query_transpose_out,
                              key_transpose_out,
                              value_transpose_out,
                              qkv_transpose_out,
                              softmax_out,
                              fmha_out,
                              gate_out,
                              config);
}

#define INSTANTIATE_GATE_ATTENTION_FORWARD(T)                    \
  template void LaunchGateAttentionMergedQKVMatmulForward<T>(    \
      const GPUContext&,                                         \
      const funcs::GateAttentionConfig<T>&,                      \
      const DenseTensor*,                                        \
      DenseTensor*,                                              \
      const DenseTensor&);                                       \
  template void LaunchGateAttentionSeparatedQKVMatmulForward<T>( \
      const GPUContext&,                                         \
      const funcs::GateAttentionConfig<T>&,                      \
      const DenseTensor*,                                        \
      const DenseTensor*,                                        \
      DenseTensor*,                                              \
      DenseTensor*,                                              \
      DenseTensor*,                                              \
      const DenseTensor&,                                        \
      const DenseTensor&,                                        \
      const DenseTensor&);                                       \
  template void LaunchGateAttentionFMHAForward<T>(               \
      const GPUContext&,                                         \
      const DenseTensor*,                                        \
      const DenseTensor*,                                        \
      DenseTensor*,                                              \
      DenseTensor*,                                              \
      DenseTensor*,                                              \
      DenseTensor*,                                              \
      DenseTensor*,                                              \
      DenseTensor*,                                              \
      DenseTensor*,                                              \
      DenseTensor*,                                              \
      funcs::GateAttentionConfig<T>*);                           \
  template void LaunchGateAttentionGatingLinearForward<T>(       \
      const GPUContext&,                                         \
      const funcs::GateAttentionConfig<T>&,                      \
      const DenseTensor*,                                        \
      const DenseTensor*,                                        \
      DenseTensor*,                                              \
      bool,                                                      \
      const DenseTensor&,                                        \
      const DenseTensor&);                                       \
  template void LaunchGateAttentionOutputLinearForward<T>(       \
      const GPUContext&,                                         \
      const funcs::GateAttentionConfig<T>&,                      \
      const DenseTensor*,                                        \
      DenseTensor*,                                              \
      bool,                                                      \
      const DenseTensor&,                                        \
      const DenseTensor&)

INSTANTIATE_GATE_ATTENTION_FORWARD(float);
INSTANTIATE_GATE_ATTENTION_FORWARD(phi::float16);
INSTANTIATE_GATE_ATTENTION_FORWARD(phi::bfloat16);
#ifndef PADDLE_WITH_HIP
INSTANTIATE_GATE_ATTENTION_FORWARD(double);
#endif

#undef INSTANTIATE_GATE_ATTENTION_FORWARD

}  // namespace fusion
}  // namespace phi
