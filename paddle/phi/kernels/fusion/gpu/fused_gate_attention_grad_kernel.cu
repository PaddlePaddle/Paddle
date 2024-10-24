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
struct SigmoidMultiplyGradFunctor {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);

  // Gradient of Multiply:
  //  dx = dout * y
  //  dy = dout * x
  // Gradient of Sigmoid: dx = dout * out * (1 - out)
  inline HOSTDEVICE phi::Array<T, 2> operator()(const T dout,
                                                const T x,
                                                T y) const {
    MPType x_mp = static_cast<MPType>(x);
    T sigmoid_out = static_cast<T>(one / (one + exp(-x_mp)));
    T d_sigmoid_out = dout * y;
    phi::Array<T, 2> outs;
    outs[0] = d_sigmoid_out * sigmoid_out *
              (static_cast<T>(1.0f) - sigmoid_out);  // dx
    outs[1] = dout * sigmoid_out;                    // dy
    return outs;
  }
};

template <typename T>
void ComputeMergedQKVMatmulBackward(
    const GPUContext &dev_ctx,
    const phi::funcs::GateAttentionGradConfig<T> &config,
    const phi::DenseTensor *query,
    const phi::DenseTensor *qkv_out_grad,
    phi::DenseTensor *query_grad,
    bool use_addto,
    const phi::DenseTensor &qkv_weight_in,
    phi::DenseTensor *qkv_weight_grad) {
  auto *qkv_weight = &qkv_weight_in;
  dev_ctx.Alloc<T>(qkv_weight_grad, qkv_weight_grad->numel() * sizeof(T));

  // Gradient of GEMM(query, qkv_weight)
  int m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int n = 3 * config.num_heads * config.head_dim;
  int k = config.q_dim;
  auto qkv_compute =
      phi::fusion::AttnMatMul<T>(dev_ctx, false, true, m, n, k, false);
  qkv_compute.ComputeBackward(query,
                              qkv_weight,
                              qkv_out_grad,
                              query_grad,
                              qkv_weight_grad,
                              nullptr,
                              use_addto);
}

template <typename T>
void ComputeSeparatedQKVMatmulBackward(
    const phi::GPUContext &dev_ctx,
    const phi::funcs::GateAttentionGradConfig<T> &config,
    const phi::DenseTensor *query,
    const phi::DenseTensor *key,
    const phi::DenseTensor *query_out_grad,
    const phi::DenseTensor *key_out_grad,
    const phi::DenseTensor *value_out_grad,
    phi::DenseTensor *query_grad,
    phi::DenseTensor *key_grad,
    bool use_addto,
    const phi::DenseTensor &query_weight_in,
    const phi::DenseTensor &key_weight_in,
    const phi::DenseTensor &value_weight_in,
    phi::DenseTensor *query_weight_grad,
    phi::DenseTensor *key_weight_grad,
    phi::DenseTensor *value_weight_grad) {
  // Gradient of GEMM(key, k_weight)
  const auto *key_weight = &key_weight_in;
  dev_ctx.Alloc<T>(key_weight_grad, key_weight_grad->numel() * sizeof(T));

  int kv_m = config.batch_size * config.seq_len_m * config.m_size;
  int kv_n = config.num_heads * config.head_dim;
  int kv_k = config.kv_dim;
  auto kv_compute = phi::fusion::AttnMatMul<T>(
      dev_ctx, false, false, kv_m, kv_n, kv_k, false);
  kv_compute.ComputeBackward(
      key, key_weight, key_out_grad, key_grad, key_weight_grad, nullptr, false);

  // Gradient of GEMM(value, v_weight)
  auto *value_weight = &value_weight_in;
  dev_ctx.Alloc<T>(value_weight_grad, value_weight_grad->numel() * sizeof(T));

  kv_compute.ComputeBackward(key,
                             value_weight,
                             value_out_grad,
                             key_grad,
                             value_weight_grad,
                             nullptr,
                             true);

  // Gradient of GEMM(query, query_weight)
  const auto *query_weight = &query_weight_in;
  dev_ctx.Alloc<T>(query_weight_grad, query_weight_grad->numel() * sizeof(T));

  int q_m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int q_n = config.num_heads * config.head_dim;
  int q_k = config.q_dim;
  auto q_compute =
      phi::fusion::AttnMatMul<T>(dev_ctx, false, false, q_m, q_n, q_k, false);
  q_compute.ComputeBackward(query,
                            query_weight,
                            query_out_grad,
                            query_grad,
                            query_weight_grad,
                            nullptr,
                            use_addto);
}

template <typename T>
void ComputeGatingLinearBackward(
    const GPUContext &dev_ctx,
    const phi::funcs::GateAttentionGradConfig<T> &config,
    const phi::DenseTensor *query,
    const phi::DenseTensor *fmha_out,
    const phi::DenseTensor *gate_out_grad,
    phi::DenseTensor *query_grad,
    phi::DenseTensor *fmha_out_grad,
    bool use_fused_matmul_bias,
    const phi::DenseTensor &gate_weight_in,
    const phi::DenseTensor &gate_bias_in,
    phi::DenseTensor *gate_weight_grad,
    phi::DenseTensor *gate_bias_grad) {
  const auto *gate_weight = &gate_weight_in;
  const auto *gate_bias = &gate_bias_in;

  // Re-compute gate_bias_out
  phi::DenseTensor gate_bias_out;
  gate_bias_out.Resize(config.gate_out_dims);
  dev_ctx.Alloc<T>(&gate_bias_out, gate_bias_out.numel() * sizeof(T));

  int m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int n = config.num_heads * config.head_dim;
  int k = config.q_dim;
  auto gate_linear =
      phi::fusion::AttnMatMul<T>(dev_ctx, false, false, m, n, k, true);
  gate_linear.ComputeForward(gate_weight,
                             query,
                             gate_bias,
                             &gate_bias_out,
                             &gate_bias_out,
                             use_fused_matmul_bias);

  // Gradient of sigmoid(gate_bias_out) * fmha_out
  // Compute inplace and save gate_bias_out_grad to gate_bias_out.
  std::vector<const phi::DenseTensor *> ins = {
      gate_out_grad, &gate_bias_out, fmha_out};
  std::vector<phi::DenseTensor *> outs = {&gate_bias_out, fmha_out_grad};
  phi::funcs::ElementwiseKernel<T, SigmoidMultiplyGradFunctor<T>, 2>(
      dev_ctx, ins, &outs, SigmoidMultiplyGradFunctor<T>());

  // Gradient of GEMM(query, gate_weight) + gate_bias
  dev_ctx.Alloc<T>(gate_weight_grad, gate_weight_grad->numel() * sizeof(T));
  dev_ctx.Alloc<T>(gate_bias_grad, gate_bias_grad->numel() * sizeof(T));

  gate_linear.ComputeBackward(query,
                              gate_weight,
                              &gate_bias_out,
                              query_grad,
                              gate_weight_grad,
                              gate_bias_grad,
                              false,
                              use_fused_matmul_bias);
}

template <typename T>
void ComputeOutputLinearBackward(
    const GPUContext &dev_ctx,
    const phi::funcs::GateAttentionGradConfig<T> &config,
    const phi::DenseTensor *input,
    phi::DenseTensor *input_grad,
    bool use_fused_matmul_bias,
    const phi::DenseTensor &out_grad_in,
    const phi::DenseTensor &out_linear_weight_in,
    phi::DenseTensor *out_linear_weight_grad,
    phi::DenseTensor *out_linear_bias_grad) {
  const auto *out_grad = &out_grad_in;
  const auto *out_linear_weight = &out_linear_weight_in;

  dev_ctx.Alloc<T>(out_linear_weight_grad,
                   out_linear_weight_grad->numel() * sizeof(T));
  dev_ctx.Alloc<T>(out_linear_bias_grad,
                   out_linear_bias_grad->numel() * sizeof(T));

  int m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int n = config.q_dim;
  int k = config.num_heads * config.head_dim;
  auto out_linear =
      phi::fusion::AttnMatMul<T>(dev_ctx, false, false, m, n, k, true);
  out_linear.ComputeBackward(input,
                             out_linear_weight,
                             out_grad,
                             input_grad,
                             out_linear_weight_grad,
                             out_linear_bias_grad,
                             false,
                             use_fused_matmul_bias);
}

template <typename T, typename Context>
void FusedGateAttentionGradKernel(
    const Context &dev_ctx,
    const DenseTensor &query_in,
    const paddle::optional<DenseTensor> &key_in,
    const paddle::optional<DenseTensor> &query_weight_in,
    const paddle::optional<DenseTensor> &key_weight_in,
    const paddle::optional<DenseTensor> &value_weight_in,
    const paddle::optional<DenseTensor> &qkv_weight_in,
    const paddle::optional<DenseTensor> &nonbatched_bias_in,
    const paddle::optional<DenseTensor> &src_mask_in,
    const paddle::optional<DenseTensor> &gate_weight_in,
    const paddle::optional<DenseTensor> &gate_bias_in,
    const DenseTensor &out_linear_weight_in,
    const DenseTensor &out_linear_bias_in,
    const paddle::optional<DenseTensor> &query_transpose_out_in,
    const paddle::optional<DenseTensor> &key_transpose_out_in,
    const paddle::optional<DenseTensor> &value_transpose_out_in,
    const paddle::optional<DenseTensor> &qkv_transpose_out_in,
    const paddle::optional<DenseTensor> &softmax_out_in,
    const paddle::optional<DenseTensor> &softmax_lse_in,
    const DenseTensor &fmha_out_in,
    const paddle::optional<DenseTensor> &gate_out_in,
    const DenseTensor &out_grad_in,
    bool has_gating,
    bool merge_qkv,
    bool use_flash_attn,
    DenseTensor *query_grad,
    DenseTensor *key_grad,
    DenseTensor *query_weight_grad,
    DenseTensor *key_weight_grad,
    DenseTensor *value_weight_grad,
    DenseTensor *qkv_weight_grad,
    DenseTensor *nonbatched_bias_grad,
    DenseTensor *gate_weight_grad,
    DenseTensor *gate_bias_grad,
    DenseTensor *out_linear_weight_grad,
    DenseTensor *out_linear_bias_grad) {
  // forward input
  const auto *query = &query_in;
  const auto *key = key_in.get_ptr();
  const auto *query_weight = query_weight_in.get_ptr();
  const auto *qkv_weight = qkv_weight_in.get_ptr();

  // forward output, backward input
  const auto *q_transpose_out = query_transpose_out_in.get_ptr();
  const auto *k_transpose_out = key_transpose_out_in.get_ptr();
  const auto *v_transpose_out = value_transpose_out_in.get_ptr();
  const auto *qkv_transpose_out = qkv_transpose_out_in.get_ptr();
  const auto *fmha_out = &fmha_out_in;
  const auto *gate_out = gate_out_in.get_ptr();

  bool use_fused_matmul_bias = true;
  phi::funcs::AllocWithDebugInfo<T>(dev_ctx, "query_grad", query_grad);

  phi::funcs::GateAttentionGradConfig<T> config(dev_ctx,
                                                query,
                                                key,
                                                query_weight,
                                                qkv_weight,
                                                merge_qkv,
                                                has_gating,
                                                use_flash_attn);

  phi::DenseTensor fmha_out_grad;
  fmha_out_grad.Resize(config.gate_out_dims);
  phi::funcs::AllocWithDebugInfo<T>(dev_ctx, "fmha_out_grad", &fmha_out_grad);
  if (has_gating) {
    // 1. Gradient of Output Linear: out = Linear(gate_out)
    phi::DenseTensor gate_out_grad;
    gate_out_grad.Resize(config.gate_out_dims);
    phi::funcs::AllocWithDebugInfo<T>(dev_ctx, "gate_out_grad", &gate_out_grad);
    ComputeOutputLinearBackward<T>(dev_ctx,
                                   config,
                                   gate_out,
                                   &gate_out_grad,
                                   use_fused_matmul_bias,
                                   out_grad_in,
                                   out_linear_weight_in,
                                   out_linear_weight_grad,
                                   out_linear_bias_grad);

    // 2. Gradient of Gating Linear
    // Forward: gate_out = Sigmoid(Linear(fmha_out)) * fmha_out
    ComputeGatingLinearBackward<T>(dev_ctx,
                                   config,
                                   query,
                                   fmha_out,
                                   &gate_out_grad,
                                   query_grad,
                                   &fmha_out_grad,
                                   use_fused_matmul_bias,
                                   gate_weight_in.get(),
                                   gate_bias_in.get(),
                                   gate_weight_grad,
                                   gate_bias_grad);
  } else {
    // 1. Gradient of Output Linear: out = Linear(fmha_grad)
    ComputeOutputLinearBackward<T>(dev_ctx,
                                   config,
                                   fmha_out,
                                   &fmha_out_grad,
                                   use_fused_matmul_bias,
                                   out_grad_in,
                                   out_linear_weight_in,
                                   out_linear_weight_grad,
                                   out_linear_bias_grad);
  }

  // 3. Gradient of FMHA
  if (nonbatched_bias_grad) {
    phi::funcs::AllocWithDebugInfo<T>(
        dev_ctx, "nonbatched_bias_grad", nonbatched_bias_grad);
  }

  if (config.CanUseFlashAttn()) {
    const auto *nonbatched_bias = nonbatched_bias_in.get_ptr();
    const auto *src_mask = src_mask_in.get_ptr();
    const auto *softmax_lse = softmax_lse_in.get_ptr();

    auto fmha_compute = phi::funcs::FlashAttnWithGating<T>(dev_ctx, merge_qkv);
    fmha_compute.ComputeBackward(qkv_transpose_out,
                                 src_mask,
                                 nonbatched_bias,
                                 softmax_lse,
                                 fmha_out,
                                 &fmha_out_grad,
                                 nullptr,
                                 nonbatched_bias_grad,
                                 &config);
  } else {
    const auto *softmax_out = softmax_out_in.get_ptr();

    auto fmha_compute = phi::funcs::FMHAGateRef<T>(dev_ctx, merge_qkv);
    fmha_compute.ComputeBackward(q_transpose_out,
                                 k_transpose_out,
                                 v_transpose_out,
                                 qkv_transpose_out,
                                 softmax_out,
                                 &fmha_out_grad,
                                 nullptr,
                                 nonbatched_bias_grad,
                                 &config);
  }

  bool use_addto = has_gating ? true : false;
  if (merge_qkv) {
    // 4. Gradient of Merged QKV Matmul
    phi::DenseTensor *qkv_out_grad = config.GetQKVOutGrad();
    ComputeMergedQKVMatmulBackward<T>(dev_ctx,
                                      config,
                                      query,
                                      qkv_out_grad,
                                      query_grad,
                                      use_addto,
                                      qkv_weight_in.get(),
                                      qkv_weight_grad);
  } else {
    // 4. Gradient of Separated QKV Matmul
    if (key_grad) {
      phi::funcs::AllocWithDebugInfo<T>(dev_ctx, "key_grad", key_grad);
    }
    phi::DenseTensor *query_out_grad = config.GetQueryOutGrad();
    phi::DenseTensor *key_out_grad = config.GetKeyOutGrad();
    phi::DenseTensor *value_out_grad = config.GetValueOutGrad();
    ComputeSeparatedQKVMatmulBackward<T>(dev_ctx,
                                         config,
                                         query,
                                         key,
                                         query_out_grad,
                                         key_out_grad,
                                         value_out_grad,
                                         query_grad,
                                         key_grad,
                                         use_addto,
                                         query_weight_in.get(),
                                         key_weight_in.get(),
                                         value_weight_in.get(),
                                         query_weight_grad,
                                         key_weight_grad,
                                         value_weight_grad);
  }
}
}  // namespace fusion
}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(fused_gate_attention_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedGateAttentionGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#else
PD_REGISTER_KERNEL(fused_gate_attention_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedGateAttentionGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif
