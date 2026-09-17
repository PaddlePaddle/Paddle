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
struct SigmoidMultiplyGradFunctor {
  using MT = typename MPTypeTrait<T>::Type;
  MT one = static_cast<MT>(1.0f);

  // Gradient of Multiply:
  //  dx = dout * y
  //  dy = dout * x
  // Gradient of Sigmoid: dx = dout * out * (1 - out)
  inline HOSTDEVICE phi::Array<T, 2> operator()(const T dout,
                                                const T x,
                                                T y) const {
    MT x_mp = static_cast<MT>(x);
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
void LaunchGateAttentionMergedQKVMatmulBackward(
    const GPUContext& dev_ctx,
    const funcs::GateAttentionGradConfig<T>& config,
    const DenseTensor* query,
    const DenseTensor* qkv_out_grad,
    DenseTensor* query_grad,
    bool use_addto,
    const DenseTensor& qkv_weight_in,
    DenseTensor* qkv_weight_grad) {
  auto* qkv_weight = &qkv_weight_in;
  dev_ctx.Alloc<T>(qkv_weight_grad, qkv_weight_grad->numel() * sizeof(T));

  // Gradient of GEMM(query, qkv_weight)
  int64_t m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int64_t n = 3 * config.num_heads * config.head_dim;
  int64_t k = config.q_dim;
  PADDLE_ENFORCE_LE_INT_MAX(m, "merged_qkv_num_tokens");
  PADDLE_ENFORCE_LE_INT_MAX(n, "merged_qkv_hidden_size");
  PADDLE_ENFORCE_LE_INT_MAX(k, "q_input_dim");
  auto qkv_compute = fusion::AttnMatMul<T>(dev_ctx,
                                           false,
                                           true,
                                           static_cast<int>(m),
                                           static_cast<int>(n),
                                           static_cast<int>(k),
                                           false);
  qkv_compute.ComputeBackward(query,
                              qkv_weight,
                              qkv_out_grad,
                              query_grad,
                              qkv_weight_grad,
                              nullptr,
                              use_addto);
}

template <typename T>
void LaunchGateAttentionSeparatedQKVMatmulBackward(
    const GPUContext& dev_ctx,
    const funcs::GateAttentionGradConfig<T>& config,
    const DenseTensor* query,
    const DenseTensor* key,
    const DenseTensor* query_out_grad,
    const DenseTensor* key_out_grad,
    const DenseTensor* value_out_grad,
    DenseTensor* query_grad,
    DenseTensor* key_grad,
    bool use_addto,
    const DenseTensor& query_weight_in,
    const DenseTensor& key_weight_in,
    const DenseTensor& value_weight_in,
    DenseTensor* query_weight_grad,
    DenseTensor* key_weight_grad,
    DenseTensor* value_weight_grad) {
  // Gradient of GEMM(key, k_weight)
  const auto* key_weight = &key_weight_in;
  dev_ctx.Alloc<T>(key_weight_grad, key_weight_grad->numel() * sizeof(T));

  int64_t kv_m = config.batch_size * config.seq_len_m * config.m_size;
  int64_t kv_n = config.num_heads * config.head_dim;
  int64_t kv_k = config.kv_dim;
  PADDLE_ENFORCE_LE_INT_MAX(kv_m, "kv_num_tokens");
  PADDLE_ENFORCE_LE_INT_MAX(kv_n, "kv_hidden_size");
  PADDLE_ENFORCE_LE_INT_MAX(kv_k, "kv_input_dim");
  auto kv_compute = fusion::AttnMatMul<T>(dev_ctx,
                                          false,
                                          false,
                                          static_cast<int>(kv_m),
                                          static_cast<int>(kv_n),
                                          static_cast<int>(kv_k),
                                          false);
  kv_compute.ComputeBackward(
      key, key_weight, key_out_grad, key_grad, key_weight_grad, nullptr, false);

  // Gradient of GEMM(value, v_weight)
  auto* value_weight = &value_weight_in;
  dev_ctx.Alloc<T>(value_weight_grad, value_weight_grad->numel() * sizeof(T));

  kv_compute.ComputeBackward(key,
                             value_weight,
                             value_out_grad,
                             key_grad,
                             value_weight_grad,
                             nullptr,
                             true);

  // Gradient of GEMM(query, query_weight)
  const auto* query_weight = &query_weight_in;
  dev_ctx.Alloc<T>(query_weight_grad, query_weight_grad->numel() * sizeof(T));

  int64_t q_m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int64_t q_n = config.num_heads * config.head_dim;
  int64_t q_k = config.q_dim;
  PADDLE_ENFORCE_LE_INT_MAX(q_m, "q_num_tokens");
  PADDLE_ENFORCE_LE_INT_MAX(q_n, "q_hidden_size");
  PADDLE_ENFORCE_LE_INT_MAX(q_k, "q_input_dim");
  auto q_compute = fusion::AttnMatMul<T>(dev_ctx,
                                         false,
                                         false,
                                         static_cast<int>(q_m),
                                         static_cast<int>(q_n),
                                         static_cast<int>(q_k),
                                         false);
  q_compute.ComputeBackward(query,
                            query_weight,
                            query_out_grad,
                            query_grad,
                            query_weight_grad,
                            nullptr,
                            use_addto);
}

template <typename T>
void LaunchGateAttentionGatingLinearBackward(
    const GPUContext& dev_ctx,
    const funcs::GateAttentionGradConfig<T>& config,
    const DenseTensor* query,
    const DenseTensor* fmha_out,
    const DenseTensor* gate_out_grad,
    DenseTensor* query_grad,
    DenseTensor* fmha_out_grad,
    bool use_fused_matmul_bias,
    const DenseTensor& gate_weight_in,
    const DenseTensor& gate_bias_in,
    DenseTensor* gate_weight_grad,
    DenseTensor* gate_bias_grad) {
  const auto* gate_weight = &gate_weight_in;
  const auto* gate_bias = &gate_bias_in;

  // Re-compute gate_bias_out
  DenseTensor gate_bias_out;
  gate_bias_out.Resize(config.gate_out_dims);
  dev_ctx.Alloc<T>(&gate_bias_out, gate_bias_out.numel() * sizeof(T));

  int64_t m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int64_t n = config.num_heads * config.head_dim;
  int64_t k = config.q_dim;
  PADDLE_ENFORCE_LE_INT_MAX(m, "gate_num_tokens");
  PADDLE_ENFORCE_LE_INT_MAX(n, "gate_hidden_size");
  PADDLE_ENFORCE_LE_INT_MAX(k, "gate_input_dim");
  auto gate_linear = fusion::AttnMatMul<T>(dev_ctx,
                                           false,
                                           false,
                                           static_cast<int>(m),
                                           static_cast<int>(n),
                                           static_cast<int>(k),
                                           true);
  gate_linear.ComputeForward(gate_weight,
                             query,
                             gate_bias,
                             &gate_bias_out,
                             &gate_bias_out,
                             use_fused_matmul_bias);

  // Gradient of sigmoid(gate_bias_out) * fmha_out
  // Compute inplace and save gate_bias_out_grad to gate_bias_out.
  std::vector<const DenseTensor*> ins = {
      gate_out_grad, &gate_bias_out, fmha_out};
  std::vector<DenseTensor*> outs = {&gate_bias_out, fmha_out_grad};
  funcs::ElementwiseKernel<T, SigmoidMultiplyGradFunctor<T>, 2>(
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
void LaunchGateAttentionOutputLinearBackward(
    const GPUContext& dev_ctx,
    const funcs::GateAttentionGradConfig<T>& config,
    const DenseTensor* input,
    DenseTensor* input_grad,
    bool use_fused_matmul_bias,
    const DenseTensor& out_grad_in,
    const DenseTensor& out_linear_weight_in,
    DenseTensor* out_linear_weight_grad,
    DenseTensor* out_linear_bias_grad) {
  const auto* out_grad = &out_grad_in;
  const auto* out_linear_weight = &out_linear_weight_in;

  dev_ctx.Alloc<T>(out_linear_weight_grad,
                   out_linear_weight_grad->numel() * sizeof(T));
  dev_ctx.Alloc<T>(out_linear_bias_grad,
                   out_linear_bias_grad->numel() * sizeof(T));

  int64_t m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int64_t n = config.q_dim;
  int64_t k = config.num_heads * config.head_dim;
  PADDLE_ENFORCE_LE_INT_MAX(m, "out_linear_num_tokens");
  PADDLE_ENFORCE_LE_INT_MAX(n, "out_hidden_size");
  PADDLE_ENFORCE_LE_INT_MAX(k, "out_linear_input_dim");
  auto out_linear = fusion::AttnMatMul<T>(dev_ctx,
                                          false,
                                          false,
                                          static_cast<int>(m),
                                          static_cast<int>(n),
                                          static_cast<int>(k),
                                          true);
  out_linear.ComputeBackward(input,
                             out_linear_weight,
                             out_grad,
                             input_grad,
                             out_linear_weight_grad,
                             out_linear_bias_grad,
                             false,
                             use_fused_matmul_bias);
}

template <typename T>
void LaunchGateAttentionFMHABackward(
    const GPUContext& dev_ctx,
    const DenseTensor* query_transpose_out,
    const DenseTensor* key_transpose_out,
    const DenseTensor* value_transpose_out,
    const DenseTensor* qkv_transpose_out,
    const DenseTensor* softmax_out,
    const DenseTensor* softmax_lse,
    const DenseTensor* src_mask,
    const DenseTensor* nonbatched_bias,
    const DenseTensor* fmha_out,
    const DenseTensor* fmha_out_grad,
    DenseTensor* nonbatched_bias_grad,
    funcs::GateAttentionGradConfig<T>* config) {
  if (config->CanUseFlashAttn()) {
    auto fmha_compute =
        funcs::FlashAttnWithGating<T>(dev_ctx, config->merge_qkv);
    fmha_compute.ComputeBackward(qkv_transpose_out,
                                 src_mask,
                                 nonbatched_bias,
                                 softmax_lse,
                                 fmha_out,
                                 fmha_out_grad,
                                 nullptr,
                                 nonbatched_bias_grad,
                                 config);
    return;
  }

  auto fmha_compute = funcs::FMHAGateRef<T>(dev_ctx, config->merge_qkv);
  fmha_compute.ComputeBackward(query_transpose_out,
                               key_transpose_out,
                               value_transpose_out,
                               qkv_transpose_out,
                               softmax_out,
                               fmha_out_grad,
                               nullptr,
                               nonbatched_bias_grad,
                               config);
}

#define INSTANTIATE_GATE_ATTENTION_BACKWARD(T)                    \
  template void LaunchGateAttentionMergedQKVMatmulBackward<T>(    \
      const GPUContext&,                                          \
      const funcs::GateAttentionGradConfig<T>&,                   \
      const DenseTensor*,                                         \
      const DenseTensor*,                                         \
      DenseTensor*,                                               \
      bool,                                                       \
      const DenseTensor&,                                         \
      DenseTensor*);                                              \
  template void LaunchGateAttentionSeparatedQKVMatmulBackward<T>( \
      const GPUContext&,                                          \
      const funcs::GateAttentionGradConfig<T>&,                   \
      const DenseTensor*,                                         \
      const DenseTensor*,                                         \
      const DenseTensor*,                                         \
      const DenseTensor*,                                         \
      const DenseTensor*,                                         \
      DenseTensor*,                                               \
      DenseTensor*,                                               \
      bool,                                                       \
      const DenseTensor&,                                         \
      const DenseTensor&,                                         \
      const DenseTensor&,                                         \
      DenseTensor*,                                               \
      DenseTensor*,                                               \
      DenseTensor*);                                              \
  template void LaunchGateAttentionGatingLinearBackward<T>(       \
      const GPUContext&,                                          \
      const funcs::GateAttentionGradConfig<T>&,                   \
      const DenseTensor*,                                         \
      const DenseTensor*,                                         \
      const DenseTensor*,                                         \
      DenseTensor*,                                               \
      DenseTensor*,                                               \
      bool,                                                       \
      const DenseTensor&,                                         \
      const DenseTensor&,                                         \
      DenseTensor*,                                               \
      DenseTensor*);                                              \
  template void LaunchGateAttentionOutputLinearBackward<T>(       \
      const GPUContext&,                                          \
      const funcs::GateAttentionGradConfig<T>&,                   \
      const DenseTensor*,                                         \
      DenseTensor*,                                               \
      bool,                                                       \
      const DenseTensor&,                                         \
      const DenseTensor&,                                         \
      DenseTensor*,                                               \
      DenseTensor*);                                              \
  template void LaunchGateAttentionFMHABackward<T>(               \
      const GPUContext&,                                          \
      const DenseTensor*,                                         \
      const DenseTensor*,                                         \
      const DenseTensor*,                                         \
      const DenseTensor*,                                         \
      const DenseTensor*,                                         \
      const DenseTensor*,                                         \
      const DenseTensor*,                                         \
      const DenseTensor*,                                         \
      const DenseTensor*,                                         \
      const DenseTensor*,                                         \
      DenseTensor*,                                               \
      funcs::GateAttentionGradConfig<T>*)

INSTANTIATE_GATE_ATTENTION_BACKWARD(float);
INSTANTIATE_GATE_ATTENTION_BACKWARD(phi::float16);
INSTANTIATE_GATE_ATTENTION_BACKWARD(phi::bfloat16);
#ifndef PADDLE_WITH_HIP
INSTANTIATE_GATE_ATTENTION_BACKWARD(double);
#endif

#undef INSTANTIATE_GATE_ATTENTION_BACKWARD

}  // namespace fusion
}  // namespace phi
