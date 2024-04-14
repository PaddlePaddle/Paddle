/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/fused/fused_gate_attention.h"
#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/fusion/gpu/attn_gemm.h"

namespace paddle {
namespace operators {

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
void ComputeMergedQKVMatmulForward(const framework::ExecutionContext &ctx,
                                   const GateAttentionConfig<T> &config,
                                   const phi::DenseTensor *query,
                                   phi::DenseTensor *qkv_out) {
  // query: shape=[batch_size, seq_len_m, seq_len_r, qkv_dim]
  // qkv_weight: shape=[3, num_heads, head_dim, qkv_dim]
  // qkv_out: shape=[batch_size, seq_len_m, seq_len_r, 3, num_heads, head_dim]
  auto *qkv_weight = ctx.Input<phi::DenseTensor>("QKVWeight");

  // qkv_out = GEMM(query, qkv_weight^T)
  int m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int n = 3 * config.num_heads * config.head_dim;
  int k = config.q_dim;
  auto qkv_compute = phi::fusion::AttnMatMul<T>(
      ctx.cuda_device_context(), false, true, m, n, k, false);
  qkv_compute.ComputeForward(qkv_weight, query, nullptr, qkv_out, nullptr);
}

template <typename T>
void ComputeMergedQKVMatmulBackward(const framework::ExecutionContext &ctx,
                                    const GateAttentionGradConfig<T> &config,
                                    const phi::DenseTensor *query,
                                    const phi::DenseTensor *qkv_out_grad,
                                    phi::DenseTensor *query_grad,
                                    bool use_addto) {
  auto *qkv_weight = ctx.Input<phi::DenseTensor>("QKVWeight");
  auto *qkv_weight_grad =
      ctx.Output<phi::DenseTensor>(framework::GradVarName("QKVWeight"));
  auto &dev_ctx = ctx.template device_context<phi::GPUContext>();
  dev_ctx.Alloc<T>(qkv_weight_grad, qkv_weight_grad->numel() * sizeof(T));

  // Gradient of GEMM(query, qkv_weight)
  int m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int n = 3 * config.num_heads * config.head_dim;
  int k = config.q_dim;
  auto qkv_compute = phi::fusion::AttnMatMul<T>(
      ctx.cuda_device_context(), false, true, m, n, k, false);
  qkv_compute.ComputeBackward(query,
                              qkv_weight,
                              qkv_out_grad,
                              query_grad,
                              qkv_weight_grad,
                              nullptr,
                              use_addto);
}

template <typename T>
void ComputeSeparatedQKVMatmulForward(const framework::ExecutionContext &ctx,
                                      const GateAttentionConfig<T> &config,
                                      const phi::DenseTensor *query,
                                      const phi::DenseTensor *key,
                                      phi::DenseTensor *query_out,
                                      phi::DenseTensor *key_out,
                                      phi::DenseTensor *value_out) {
  auto *query_weight = ctx.Input<phi::DenseTensor>("QueryWeight");
  auto *key_weight = ctx.Input<phi::DenseTensor>("KeyWeight");
  auto *value_weight = ctx.Input<phi::DenseTensor>("ValueWeight");

  // query_out = GEMM(query, query_weight)
  // query: shape=[batch_size, seq_len_m, seq_len_r, q_dim]
  // query_weight: shape=[q_dim, num_heads, head_dim]
  // query_out: shape=[batch_size, seq_len_m, seq_len_r, num_heads, head_dim]
  int q_m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int q_n = config.num_heads * config.head_dim;
  int q_k = config.q_dim;
  auto q_compute = phi::fusion::AttnMatMul<T>(
      ctx.cuda_device_context(), false, false, q_m, q_n, q_k, false);
  q_compute.ComputeForward(query_weight, query, nullptr, query_out, nullptr);

  // k_out = GEMM(key, key_weight)
  // key: shape=[batch_size, seq_len_m, m_size, kv_dim]
  // key_weight: shape=[kv_dim, num_heads, head_dim]
  // key_out: shape=[batch_size, seq_len_m, m_size, num_heads, head_dim]
  int kv_m = config.batch_size * config.seq_len_m * config.m_size;
  int kv_n = config.num_heads * config.head_dim;
  int kv_k = config.kv_dim;
  auto kv_compute = phi::fusion::AttnMatMul<T>(
      ctx.cuda_device_context(), false, false, kv_m, kv_n, kv_k, false);
  kv_compute.ComputeForward(key_weight, key, nullptr, key_out, nullptr);

  // value_out = GEMM(value, value_weight)
  kv_compute.ComputeForward(value_weight, key, nullptr, value_out, nullptr);
}

template <typename T>
void ComputeSeparatedQKVMatmulBackward(const framework::ExecutionContext &ctx,
                                       const GateAttentionGradConfig<T> &config,
                                       const phi::DenseTensor *query,
                                       const phi::DenseTensor *key,
                                       const phi::DenseTensor *query_out_grad,
                                       const phi::DenseTensor *key_out_grad,
                                       const phi::DenseTensor *value_out_grad,
                                       phi::DenseTensor *query_grad,
                                       phi::DenseTensor *key_grad,
                                       bool use_addto) {
  // Gradient of GEMM(key, k_weight)
  const auto *key_weight = ctx.Input<phi::DenseTensor>("KeyWeight");
  auto *key_weight_grad =
      ctx.Output<phi::DenseTensor>(framework::GradVarName("KeyWeight"));
  auto &dev_ctx = ctx.template device_context<phi::GPUContext>();
  dev_ctx.Alloc<T>(key_weight_grad, key_weight_grad->numel() * sizeof(T));

  int kv_m = config.batch_size * config.seq_len_m * config.m_size;
  int kv_n = config.num_heads * config.head_dim;
  int kv_k = config.kv_dim;
  auto kv_compute = phi::fusion::AttnMatMul<T>(
      ctx.cuda_device_context(), false, false, kv_m, kv_n, kv_k, false);
  kv_compute.ComputeBackward(
      key, key_weight, key_out_grad, key_grad, key_weight_grad, nullptr, false);

  // Gradient of GEMM(value, v_weight)
  auto *value_weight = ctx.Input<phi::DenseTensor>("ValueWeight");
  auto *value_weight_grad =
      ctx.Output<phi::DenseTensor>(framework::GradVarName("ValueWeight"));
  dev_ctx.Alloc<T>(value_weight_grad, value_weight_grad->numel() * sizeof(T));

  kv_compute.ComputeBackward(key,
                             value_weight,
                             value_out_grad,
                             key_grad,
                             value_weight_grad,
                             nullptr,
                             true);

  // Gradient of GEMM(query, query_weight)
  const auto *query_weight = ctx.Input<phi::DenseTensor>("QueryWeight");
  auto *query_weight_grad =
      ctx.Output<phi::DenseTensor>(framework::GradVarName("QueryWeight"));
  dev_ctx.Alloc<T>(query_weight_grad, query_weight_grad->numel() * sizeof(T));

  int q_m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int q_n = config.num_heads * config.head_dim;
  int q_k = config.q_dim;
  auto q_compute = phi::fusion::AttnMatMul<T>(
      ctx.cuda_device_context(), false, false, q_m, q_n, q_k, false);
  q_compute.ComputeBackward(query,
                            query_weight,
                            query_out_grad,
                            query_grad,
                            query_weight_grad,
                            nullptr,
                            use_addto);
}

template <typename T>
void ComputeGatingLinearForward(const framework::ExecutionContext &ctx,
                                const GateAttentionConfig<T> &config,
                                const phi::DenseTensor *query,
                                const phi::DenseTensor *fmha_out,
                                phi::DenseTensor *gate_bias_out,
                                bool use_fused_matmul_bias) {
  auto *gate_weight = ctx.Input<phi::DenseTensor>("GateWeight");
  auto *gate_bias = ctx.Input<phi::DenseTensor>("GateBias");

  // The first gate_bias_out stores the result of the multiplication,
  // and the second gate_bias_out stores the result of the multiplication +
  // bias.
  //   gate_out = GEMM(query, gate_weight) + gate_bias
  int m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int n = config.num_heads * config.head_dim;
  int k = config.q_dim;
  auto gate_linear = phi::fusion::AttnMatMul<T>(
      ctx.cuda_device_context(), false, false, m, n, k, true);
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
      ctx.cuda_device_context(), ins, &outs, SigmoidMultiplyFunctor<T>());
}

template <typename T>
void ComputeGatingLinearBackward(const framework::ExecutionContext &ctx,
                                 const GateAttentionGradConfig<T> &config,
                                 const phi::DenseTensor *query,
                                 const phi::DenseTensor *fmha_out,
                                 const phi::DenseTensor *gate_out_grad,
                                 phi::DenseTensor *query_grad,
                                 phi::DenseTensor *fmha_out_grad,
                                 bool use_fused_matmul_bias) {
  const auto *gate_weight = ctx.Input<phi::DenseTensor>("GateWeight");
  const auto *gate_bias = ctx.Input<phi::DenseTensor>("GateBias");
  auto &dev_ctx = ctx.template device_context<phi::GPUContext>();

  // Re-compute gate_bias_out
  phi::DenseTensor gate_bias_out;
  gate_bias_out.Resize(config.gate_out_dims);
  dev_ctx.Alloc<T>(&gate_bias_out, gate_bias_out.numel() * sizeof(T));

  int m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int n = config.num_heads * config.head_dim;
  int k = config.q_dim;
  auto gate_linear = phi::fusion::AttnMatMul<T>(
      ctx.cuda_device_context(), false, false, m, n, k, true);
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
      ctx.cuda_device_context(), ins, &outs, SigmoidMultiplyGradFunctor<T>());

  // Gradient of GEMM(query, gate_weight) + gate_bias
  auto *gate_weight_grad =
      ctx.Output<phi::DenseTensor>(framework::GradVarName("GateWeight"));
  auto *gate_bias_grad =
      ctx.Output<phi::DenseTensor>(framework::GradVarName("GateBias"));
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
void ComputeOutputLinearForward(const framework::ExecutionContext &ctx,
                                const GateAttentionConfig<T> &config,
                                const phi::DenseTensor *fmha_or_gate_out,
                                phi::DenseTensor *out,
                                bool use_fused_matmul_bias) {
  const auto *out_linear_weight =
      ctx.Input<phi::DenseTensor>("OutLinearWeight");
  const auto *out_linear_bias = ctx.Input<phi::DenseTensor>("OutLinearBias");

  // out = GEMM(fmha_or_gate_out, out_linear_weight) + out_linear_bias
  int m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int n = config.q_dim;
  int k = config.num_heads * config.head_dim;
  auto out_linear = phi::fusion::AttnMatMul<T>(
      ctx.cuda_device_context(), false, false, m, n, k, true);
  out_linear.ComputeForward(out_linear_weight,
                            fmha_or_gate_out,
                            out_linear_bias,
                            out,
                            out,
                            use_fused_matmul_bias);
}

template <typename T>
void ComputeOutputLinearBackward(const framework::ExecutionContext &ctx,
                                 const GateAttentionGradConfig<T> &config,
                                 const phi::DenseTensor *input,
                                 phi::DenseTensor *input_grad,
                                 bool use_fused_matmul_bias) {
  auto &dev_ctx = ctx.template device_context<phi::GPUContext>();
  const auto *out_grad =
      ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
  const auto *out_linear_weight =
      ctx.Input<phi::DenseTensor>("OutLinearWeight");

  auto *out_linear_weight_grad =
      ctx.Output<phi::DenseTensor>(framework::GradVarName("OutLinearWeight"));
  auto *out_linear_bias_grad =
      ctx.Output<phi::DenseTensor>(framework::GradVarName("OutLinearBias"));

  dev_ctx.Alloc<T>(out_linear_weight_grad,
                   out_linear_weight_grad->numel() * sizeof(T));
  dev_ctx.Alloc<T>(out_linear_bias_grad,
                   out_linear_bias_grad->numel() * sizeof(T));

  int m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int n = config.q_dim;
  int k = config.num_heads * config.head_dim;
  auto out_linear = phi::fusion::AttnMatMul<T>(
      ctx.cuda_device_context(), false, false, m, n, k, true);
  out_linear.ComputeBackward(input,
                             out_linear_weight,
                             out_grad,
                             input_grad,
                             out_linear_weight_grad,
                             out_linear_bias_grad,
                             false,
                             use_fused_matmul_bias);
}

template <typename T, typename DeviceContext>
class FusedGateAttentionOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *query = ctx.Input<phi::DenseTensor>("Query");
    const auto *key = ctx.Input<phi::DenseTensor>("Key");
    const auto *query_weight = ctx.Input<phi::DenseTensor>("QueryWeight");
    const auto *qkv_weight = ctx.Input<phi::DenseTensor>("QKVWeight");

    const auto *src_mask = ctx.Input<phi::DenseTensor>("SrcMask");
    const auto *nonbatched_bias = ctx.Input<phi::DenseTensor>("NonbatchedBias");

    auto *q_transpose_out = ctx.Output<phi::DenseTensor>("QueryTransposeOut");
    auto *k_transpose_out = ctx.Output<phi::DenseTensor>("KeyTransposeOut");
    auto *v_transpose_out = ctx.Output<phi::DenseTensor>("ValueTransposeOut");
    auto *qkv_transpose_out = ctx.Output<phi::DenseTensor>("QKVTransposeOut");

    auto *fmha_out = ctx.Output<phi::DenseTensor>("FMHAOut");
    auto *gate_out = ctx.Output<phi::DenseTensor>("GateOut");
    auto *out = ctx.Output<phi::DenseTensor>("Out");

    const bool merge_qkv = ctx.Attr<bool>("merge_qkv");
    const bool has_gating = ctx.Attr<bool>("has_gating");
    const bool use_flash_attn = ctx.Attr<bool>("use_flash_attn");

    bool use_fused_matmul_bias = true;
    auto &dev_ctx = ctx.template device_context<phi::GPUContext>();
    AllocWithDebugInfo<T>(dev_ctx, "fmha_out", fmha_out);
    if (has_gating) {
      AllocWithDebugInfo<T>(dev_ctx, "gate_out", gate_out);
    }
    AllocWithDebugInfo<T>(dev_ctx, "out", out);

    // When seq_len_r = m_size, q_dim = kv_dim, QKV matmul can be merged.
    GateAttentionConfig<T> config(dev_ctx,
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
          phi::errors::InvalidArgument(
              "key is expected to be nullptr or the same as "
              "query, but received key=%p, query=%p.",
              key,
              query));

      // 1. Merged QKV Matmul: einsum(nbhqk,nbkhc -> nbqhc)
      phi::DenseTensor *qkv_out = config.GetQKVOut();
      ComputeMergedQKVMatmulForward<T>(ctx, config, query, qkv_out);

      if (config.CanUseFlashAttn()) {
        qkv_transpose_out->Resize(common::make_ddim({3,
                                                     config.batch_size,
                                                     config.seq_len_m,
                                                     config.seq_len_r,
                                                     config.num_heads,
                                                     config.head_dim}));
      }
      AllocWithDebugInfo<T>(dev_ctx, "qkv_transpose_out", qkv_transpose_out);
    } else {
      // 1. Separated QKV Matmul
      phi::DenseTensor *query_out = config.GetQueryOut();
      phi::DenseTensor *key_out = config.GetKeyOut();
      phi::DenseTensor *value_out = config.GetValueOut();
      ComputeSeparatedQKVMatmulForward<T>(
          ctx, config, query, key, query_out, key_out, value_out);

      AllocWithDebugInfo<T>(dev_ctx, "q_transpose_out", q_transpose_out);
      AllocWithDebugInfo<T>(dev_ctx, "k_transpose_out", k_transpose_out);
      AllocWithDebugInfo<T>(dev_ctx, "v_transpose_out", v_transpose_out);
    }

    // 2. FMHA
    if (config.CanUseFlashAttn()) {
      auto *softmax_lse = ctx.Output<phi::DenseTensor>("SoftmaxLse");
      auto fmha_compute = FlashAttnWithGating<T>(dev_ctx, merge_qkv);
      fmha_compute.ComputeForward(nonbatched_bias,
                                  src_mask,
                                  qkv_transpose_out,
                                  softmax_lse,
                                  fmha_out,
                                  &config);
    } else {
      auto *softmax_out = ctx.Output<phi::DenseTensor>("SoftmaxOut");
      AllocWithDebugInfo<T>(dev_ctx, "softmax_out", softmax_out);

      auto fmha_compute = FMHAGateRef<T>(dev_ctx, merge_qkv);
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
      ComputeGatingLinearForward<T>(
          ctx, config, query, fmha_out, gate_out, use_fused_matmul_bias);
    }

    // 4. Output Linear
    phi::DenseTensor *fmha_or_gate_out = has_gating ? gate_out : fmha_out;
    ComputeOutputLinearForward<T>(
        ctx, config, fmha_or_gate_out, out, use_fused_matmul_bias);
  }
};

template <typename T, typename DeviceContext>
class FusedGateAttentionGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    // forward input
    const auto *query = ctx.Input<phi::DenseTensor>("Query");
    const auto *key = ctx.Input<phi::DenseTensor>("Key");
    const auto *query_weight = ctx.Input<phi::DenseTensor>("QueryWeight");
    const auto *qkv_weight = ctx.Input<phi::DenseTensor>("QKVWeight");

    // forward output, backward input
    const auto *q_transpose_out =
        ctx.Input<phi::DenseTensor>("QueryTransposeOut");
    const auto *k_transpose_out =
        ctx.Input<phi::DenseTensor>("KeyTransposeOut");
    const auto *v_transpose_out =
        ctx.Input<phi::DenseTensor>("ValueTransposeOut");
    const auto *qkv_transpose_out =
        ctx.Input<phi::DenseTensor>("QKVTransposeOut");
    const auto *fmha_out = ctx.Input<phi::DenseTensor>("FMHAOut");
    const auto *gate_out = ctx.Input<phi::DenseTensor>("GateOut");

    // backward output
    auto *query_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Query"));
    auto *nonbatched_bias_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("NonbatchedBias"));

    bool has_gating = ctx.Attr<bool>("has_gating");
    bool merge_qkv = ctx.Attr<bool>("merge_qkv");
    bool use_flash_attn = ctx.Attr<bool>("use_flash_attn");

    bool use_fused_matmul_bias = true;
    auto &dev_ctx = ctx.template device_context<phi::GPUContext>();
    AllocWithDebugInfo<T>(dev_ctx, "query_grad", query_grad);

    GateAttentionGradConfig<T> config(dev_ctx,
                                      query,
                                      key,
                                      query_weight,
                                      qkv_weight,
                                      merge_qkv,
                                      has_gating,
                                      use_flash_attn);

    phi::DenseTensor fmha_out_grad;
    fmha_out_grad.Resize(config.gate_out_dims);
    AllocWithDebugInfo<T>(dev_ctx, "fmha_out_grad", &fmha_out_grad);
    if (has_gating) {
      // 1. Gradient of Output Linear: out = Linear(gate_out)
      phi::DenseTensor gate_out_grad;
      gate_out_grad.Resize(config.gate_out_dims);
      AllocWithDebugInfo<T>(dev_ctx, "gate_out_grad", &gate_out_grad);
      ComputeOutputLinearBackward<T>(
          ctx, config, gate_out, &gate_out_grad, use_fused_matmul_bias);

      // 2. Gradient of Gating Linear
      // Forward: gate_out = Sigmoid(Linear(fmha_out)) * fmha_out
      ComputeGatingLinearBackward<T>(ctx,
                                     config,
                                     query,
                                     fmha_out,
                                     &gate_out_grad,
                                     query_grad,
                                     &fmha_out_grad,
                                     use_fused_matmul_bias);
    } else {
      // 1. Gradient of Output Linear: out = Linear(fmha_grad)
      ComputeOutputLinearBackward<T>(
          ctx, config, fmha_out, &fmha_out_grad, use_fused_matmul_bias);
    }

    // 3. Gradient of FMHA
    if (nonbatched_bias_grad) {
      AllocWithDebugInfo<T>(
          dev_ctx, "nonbatched_bias_grad", nonbatched_bias_grad);
    }

    if (config.CanUseFlashAttn()) {
      const auto *nonbatched_bias =
          ctx.Input<phi::DenseTensor>("NonbatchedBias");
      const auto *src_mask = ctx.Input<phi::DenseTensor>("SrcMask");
      const auto *softmax_lse = ctx.Input<phi::DenseTensor>("SoftmaxLse");

      auto fmha_compute = FlashAttnWithGating<T>(dev_ctx, merge_qkv);
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
      const auto *softmax_out = ctx.Input<phi::DenseTensor>("SoftmaxOut");

      auto fmha_compute = FMHAGateRef<T>(dev_ctx, merge_qkv);
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
      ComputeMergedQKVMatmulBackward<T>(
          ctx, config, query, qkv_out_grad, query_grad, use_addto);
    } else {
      // 4. Gradient of Separated QKV Matmul
      auto *key_grad =
          ctx.Output<phi::DenseTensor>(framework::GradVarName("Key"));
      if (key_grad) {
        AllocWithDebugInfo<T>(dev_ctx, "key_grad", key_grad);
      }
      phi::DenseTensor *query_out_grad = config.GetQueryOutGrad();
      phi::DenseTensor *key_out_grad = config.GetKeyOutGrad();
      phi::DenseTensor *value_out_grad = config.GetValueOutGrad();
      ComputeSeparatedQKVMatmulBackward<T>(ctx,
                                           config,
                                           query,
                                           key,
                                           query_out_grad,
                                           key_out_grad,
                                           value_out_grad,
                                           query_grad,
                                           key_grad,
                                           use_addto);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
#ifdef PADDLE_WITH_HIP
PD_REGISTER_STRUCT_KERNEL(fused_gate_attention,
                          GPU,
                          ALL_LAYOUT,
                          ops::FusedGateAttentionOpKernel,
                          float,
                          phi::dtype::float16,
                          plat::bfloat16) {}
PD_REGISTER_STRUCT_KERNEL(fused_gate_attention_grad,
                          GPU,
                          ALL_LAYOUT,
                          ops::FusedGateAttentionGradKernel,
                          float,
                          phi::dtype::float16,
                          plat::bfloat16) {}
#else
PD_REGISTER_STRUCT_KERNEL(fused_gate_attention,
                          GPU,
                          ALL_LAYOUT,
                          ops::FusedGateAttentionOpKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          plat::bfloat16) {}
PD_REGISTER_STRUCT_KERNEL(fused_gate_attention_grad,
                          GPU,
                          ALL_LAYOUT,
                          ops::FusedGateAttentionGradKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          plat::bfloat16) {}
#endif
