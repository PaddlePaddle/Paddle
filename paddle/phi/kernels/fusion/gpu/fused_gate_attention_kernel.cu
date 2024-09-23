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
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/fused_gate_attention.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/fusion/gpu/attn_gemm.h"
#include "paddle/utils/optional.h"

namespace phi {
namespace fusion {

template <typename T>
struct SigmoidMultiplyFunctor {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);

  // sigmoid(x) = 1 / (1 + exp(-x))
  // out = sigmoid(x) * y
  inline HOSTDEVICE T operator()(T x, T y) const {
    MPType x_mp = static_cast<MPType>(x);
    T sigmoid_out = static_cast<T>(one / (one + exp(-x_mp)));
    return sigmoid_out * y;
  }
};

template <typename T>
void ComputeMergedQKVMatmulForward(
    const GPUContext &dev_ctx,
    const phi::funcs::GateAttentionConfig<T> &config,
    const phi::DenseTensor *query,
    phi::DenseTensor *qkv_out,
    const phi::DenseTensor &qkv_weight_in) {
  // query: shape=[batch_size, seq_len_m, seq_len_r, qkv_dim]
  // qkv_weight: shape=[3, num_heads, head_dim, qkv_dim]
  // qkv_out: shape=[batch_size, seq_len_m, seq_len_r, 3, num_heads, head_dim]
  auto *qkv_weight = &qkv_weight_in;

  // qkv_out = GEMM(query, qkv_weight^T)
  int m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int n = 3 * config.num_heads * config.head_dim;
  int k = config.q_dim;
  auto qkv_compute =
      phi::fusion::AttnMatMul<T>(dev_ctx, false, true, m, n, k, false);
  qkv_compute.ComputeForward(qkv_weight, query, nullptr, qkv_out, nullptr);
}

template <typename T>
void ComputeSeparatedQKVMatmulForward(
    const GPUContext &dev_ctx,
    const phi::funcs::GateAttentionConfig<T> &config,
    const phi::DenseTensor *query,
    const phi::DenseTensor *key,
    phi::DenseTensor *query_out,
    phi::DenseTensor *key_out,
    phi::DenseTensor *value_out,
    const phi::DenseTensor &query_weight_in,
    const phi::DenseTensor &key_weight_in,
    const phi::DenseTensor &value_weight_in) {
  auto *query_weight = &query_weight_in;
  auto *key_weight = &key_weight_in;
  auto *value_weight = &value_weight_in;

  // query_out = GEMM(query, query_weight)
  // query: shape=[batch_size, seq_len_m, seq_len_r, q_dim]
  // query_weight: shape=[q_dim, num_heads, head_dim]
  // query_out: shape=[batch_size, seq_len_m, seq_len_r, num_heads, head_dim]
  int q_m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int q_n = config.num_heads * config.head_dim;
  int q_k = config.q_dim;
  auto q_compute =
      phi::fusion::AttnMatMul<T>(dev_ctx, false, false, q_m, q_n, q_k, false);
  q_compute.ComputeForward(query_weight, query, nullptr, query_out, nullptr);

  // k_out = GEMM(key, key_weight)
  // key: shape=[batch_size, seq_len_m, m_size, kv_dim]
  // key_weight: shape=[kv_dim, num_heads, head_dim]
  // key_out: shape=[batch_size, seq_len_m, m_size, num_heads, head_dim]
  int kv_m = config.batch_size * config.seq_len_m * config.m_size;
  int kv_n = config.num_heads * config.head_dim;
  int kv_k = config.kv_dim;
  auto kv_compute = phi::fusion::AttnMatMul<T>(
      dev_ctx, false, false, kv_m, kv_n, kv_k, false);
  kv_compute.ComputeForward(key_weight, key, nullptr, key_out, nullptr);

  // value_out = GEMM(value, value_weight)
  kv_compute.ComputeForward(value_weight, key, nullptr, value_out, nullptr);
}

template <typename T>
void ComputeGatingLinearForward(
    const GPUContext &dev_ctx,
    const phi::funcs::GateAttentionConfig<T> &config,
    const phi::DenseTensor *query,
    const phi::DenseTensor *fmha_out,
    phi::DenseTensor *gate_bias_out,
    bool use_fused_matmul_bias,
    const phi::DenseTensor &gate_weight_in,
    const phi::DenseTensor &gate_bias_in) {
  auto *gate_weight = &gate_weight_in;
  auto *gate_bias = &gate_bias_in;

  // The first gate_bias_out stores the result of the multiplication,
  // and the second gate_bias_out stores the result of the multiplication +
  // bias.
  //   gate_out = GEMM(query, gate_weight) + gate_bias
  int m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int n = config.num_heads * config.head_dim;
  int k = config.q_dim;
  auto gate_linear =
      phi::fusion::AttnMatMul<T>(dev_ctx, false, false, m, n, k, true);
  gate_linear.ComputeForward(gate_weight,
                             query,
                             gate_bias,
                             gate_bias_out,
                             gate_bias_out,
                             use_fused_matmul_bias);

  // gate_out = sigmoid(gate_out) * fmha_out
  std::vector<const phi::DenseTensor *> ins = {gate_bias_out, fmha_out};
  std::vector<phi::DenseTensor *> outs = {gate_bias_out};
  phi::funcs::ElementwiseKernel<T>(
      dev_ctx, ins, &outs, SigmoidMultiplyFunctor<T>());
}

template <typename T>
void ComputeOutputLinearForward(
    const GPUContext &dev_ctx,
    const phi::funcs::GateAttentionConfig<T> &config,
    const phi::DenseTensor *fmha_or_gate_out,
    phi::DenseTensor *out,
    bool use_fused_matmul_bias,
    const phi::DenseTensor &out_linear_weight_in,
    const phi::DenseTensor &out_linear_bias_in) {
  const auto *out_linear_weight = &out_linear_weight_in;
  const auto *out_linear_bias = &out_linear_bias_in;

  // out = GEMM(fmha_or_gate_out, out_linear_weight) + out_linear_bias
  int m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int n = config.q_dim;
  int k = config.num_heads * config.head_dim;
  auto out_linear =
      phi::fusion::AttnMatMul<T>(dev_ctx, false, false, m, n, k, true);
  out_linear.ComputeForward(out_linear_weight,
                            fmha_or_gate_out,
                            out_linear_bias,
                            out,
                            out,
                            use_fused_matmul_bias);
}

template <typename T, typename Context>
void FusedGateAttentionOpKernel(
    const Context &dev_ctx,
    const DenseTensor &query_in,
    const paddle::optional<DenseTensor> &key_in,
    const paddle::optional<DenseTensor> &query_weight_in,
    const paddle::optional<DenseTensor> &key_weight_in,
    const paddle::optional<DenseTensor> &value_weight_in,
    const paddle::optional<DenseTensor> &qkv_weight_in,
    const paddle::optional<DenseTensor> &nonbatched_bias_in,
    const DenseTensor &src_mask_in,
    const paddle::optional<DenseTensor> &gate_weight_in,
    const paddle::optional<DenseTensor> &gate_bias_in,
    const DenseTensor &out_linear_weight_in,
    const DenseTensor &out_linear_bias_in,
    bool has_gating,
    bool merge_qkv,
    bool use_flash_attn,
    DenseTensor *query_transpose_out,
    DenseTensor *key_transpose_out,
    DenseTensor *value_transpose_out,
    DenseTensor *qkv_transpose_out,
    DenseTensor *softmax_out,
    DenseTensor *softmax_lse,
    DenseTensor *fmha_out,
    DenseTensor *gate_out,
    DenseTensor *out) {
  const auto *query = &query_in;
  const auto *key = key_in.get_ptr();
  const auto *query_weight = query_weight_in.get_ptr();
  const auto *qkv_weight = qkv_weight_in.get_ptr();

  const auto *src_mask = &src_mask_in;
  const auto *nonbatched_bias = nonbatched_bias_in.get_ptr();

  auto *q_transpose_out = query_transpose_out;
  auto *k_transpose_out = key_transpose_out;
  auto *v_transpose_out = value_transpose_out;

  bool use_fused_matmul_bias = true;
  phi::funcs::AllocWithDebugInfo<T>(dev_ctx, "fmha_out", fmha_out);
  if (has_gating) {
    phi::funcs::AllocWithDebugInfo<T>(dev_ctx, "gate_out", gate_out);
  }
  phi::funcs::AllocWithDebugInfo<T>(dev_ctx, "out", out);

  // When seq_len_r = m_size, q_dim = kv_dim, QKV matmul can be merged.
  phi::funcs::GateAttentionConfig<T> config(dev_ctx,
                                            query,
                                            key,
                                            query_weight,
                                            qkv_weight,
                                            merge_qkv,
                                            has_gating,
                                            use_flash_attn);

  if (merge_qkv) {
    PADDLE_ENFORCE_EQ(
        !key || query == key || query->data<T>() == key->data<T>(),
        true,
        errors::InvalidArgument("key is expected to be nullptr or the same as "
                                "query, but received key=%p, query=%p.",
                                key,
                                query));

    // 1. Merged QKV Matmul: einsum(nbhqk,nbkhc -> nbqhc)
    phi::DenseTensor *qkv_out = config.GetQKVOut();
    ComputeMergedQKVMatmulForward<T>(
        dev_ctx, config, query, qkv_out, qkv_weight_in.get());

    if (config.CanUseFlashAttn()) {
      qkv_transpose_out->Resize(common::make_ddim({3,
                                                   config.batch_size,
                                                   config.seq_len_m,
                                                   config.seq_len_r,
                                                   config.num_heads,
                                                   config.head_dim}));
    }
    phi::funcs::AllocWithDebugInfo<T>(
        dev_ctx, "qkv_transpose_out", qkv_transpose_out);
  } else {
    // 1. Separated QKV Matmul
    phi::DenseTensor *query_out = config.GetQueryOut();
    phi::DenseTensor *key_out = config.GetKeyOut();
    phi::DenseTensor *value_out = config.GetValueOut();
    ComputeSeparatedQKVMatmulForward<T>(dev_ctx,
                                        config,
                                        query,
                                        key,
                                        query_out,
                                        key_out,
                                        value_out,
                                        query_weight_in.get(),
                                        key_weight_in.get(),
                                        value_weight_in.get());

    phi::funcs::AllocWithDebugInfo<T>(
        dev_ctx, "q_transpose_out", q_transpose_out);
    phi::funcs::AllocWithDebugInfo<T>(
        dev_ctx, "k_transpose_out", k_transpose_out);
    phi::funcs::AllocWithDebugInfo<T>(
        dev_ctx, "v_transpose_out", v_transpose_out);
  }

  // 2. FMHA
  if (config.CanUseFlashAttn()) {
    auto fmha_compute = phi::funcs::FlashAttnWithGating<T>(dev_ctx, merge_qkv);
    fmha_compute.ComputeForward(nonbatched_bias,
                                src_mask,
                                qkv_transpose_out,
                                softmax_lse,
                                fmha_out,
                                &config);
  } else {
    phi::funcs::AllocWithDebugInfo<T>(dev_ctx, "softmax_out", softmax_out);

    auto fmha_compute = phi::funcs::FMHAGateRef<T>(dev_ctx, merge_qkv);
    fmha_compute.ComputeForward(nonbatched_bias,
                                src_mask,
                                q_transpose_out,
                                k_transpose_out,
                                v_transpose_out,
                                qkv_transpose_out,
                                softmax_out,
                                fmha_out,
                                gate_out,
                                &config);
  }

  // 3. Gating Linear
  if (has_gating) {
    ComputeGatingLinearForward<T>(dev_ctx,
                                  config,
                                  query,
                                  fmha_out,
                                  gate_out,
                                  use_fused_matmul_bias,
                                  gate_weight_in.get(),
                                  gate_bias_in.get());
  }

  // 4. Output Linear
  phi::DenseTensor *fmha_or_gate_out = has_gating ? gate_out : fmha_out;
  ComputeOutputLinearForward<T>(dev_ctx,
                                config,
                                fmha_or_gate_out,
                                out,
                                use_fused_matmul_bias,
                                out_linear_weight_in,
                                out_linear_bias_in);
}
}  // namespace fusion
}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(fused_gate_attention,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedGateAttentionOpKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#else
PD_REGISTER_KERNEL(fused_gate_attention,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedGateAttentionOpKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif
