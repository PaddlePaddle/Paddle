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
#include "paddle/fluid/operators/fused/attn_gemm.h"
#include "paddle/fluid/operators/fused/fused_gate_attention.h"
#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

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
  inline HOSTDEVICE phi::Array<T, 2> operator()(const T dout, const T x,
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
                                   const Tensor *query, Tensor *qkv_out) {
  // query: shape=[batch_size, seq_len_m, seq_len_r, qkv_dim]
  // qkv_weight: shape=[3, num_heads, key_dim, qkv_dim]
  // qkv_out: shape=[batch_size, seq_len_m, seq_len_r, 3, num_heads, key_dim]
  auto *qkv_weight = ctx.Input<Tensor>("QKVWeight");

  // qkv_out = GEMM(query, qkv_weight^T)
  int m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int n = 3 * config.num_heads * config.key_dim;
  int k = config.q_dim;
  auto qkv_compute =
      AttnMatMul<T>(ctx.cuda_device_context(), false, true, m, n, k, false);
  qkv_compute.ComputeForward(qkv_weight, query, nullptr, qkv_out, nullptr);
}

template <typename T>
Tensor *ComputeMergedQKVMatmulBackward(const framework::ExecutionContext &ctx,
                                       const GateAttentionGradConfig<T> &config,
                                       const Tensor *query,
                                       const Tensor *qkv_out_grad,
                                       Tensor *query_grad, bool use_addto) {
  auto *qkv_weight = ctx.Input<Tensor>("QKVWeight");
  auto *qkv_weight_grad =
      ctx.Output<Tensor>(framework::GradVarName("QKVWeight"));
  qkv_weight_grad->mutable_data<T>(ctx.GetPlace());

  // Gradient of GEMM(query, qkv_weight)
  int m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int n = 3 * config.num_heads * config.key_dim;
  int k = config.q_dim;
  auto qkv_compute =
      AttnMatMul<T>(ctx.cuda_device_context(), false, true, m, n, k, false);
  qkv_compute.ComputeBackward(query, qkv_weight, qkv_out_grad, query_grad,
                              qkv_weight_grad, nullptr, use_addto);
  return query_grad;
}

template <typename T>
void ComputeSeparatedQKVMatmulForward(const framework::ExecutionContext &ctx,
                                      const GateAttentionConfig<T> &config,
                                      const Tensor *query, const Tensor *key,
                                      Tensor *query_out, Tensor *key_out,
                                      Tensor *value_out) {
  auto *query_weight = ctx.Input<Tensor>("QueryWeight");
  auto *key_weight = ctx.Input<Tensor>("KeyWeight");
  auto *value_weight = ctx.Input<Tensor>("ValueWeight");

  // query_out = GEMM(query, query_weight)
  // query: shape=[batch_size, seq_len_m, seq_len_r, q_dim]
  // query_weight: shape=[q_dim, num_heads, key_dim]
  // query_out: shape=[batch_size, seq_len_m, seq_len_r, num_heads, key_dim]
  int q_m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int q_n = config.num_heads * config.key_dim;
  int q_k = config.q_dim;
  auto q_compute = AttnMatMul<T>(ctx.cuda_device_context(), false, false, q_m,
                                 q_n, q_k, false);
  q_compute.ComputeForward(query_weight, query, nullptr, query_out, nullptr);

  // k_out = GEMM(key, key_weight)
  // key: shape=[batch_size, seq_len_m, m_size, kv_dim]
  // key_weight: shape=[kv_dim, num_heads, key_dim]
  // key_out: shape=[batch_size, seq_len_m, m_size, num_heads, key_dim]
  int kv_m = config.batch_size * config.seq_len_m * config.m_size;
  int kv_n = config.num_heads * config.key_dim;
  int kv_k = config.kv_dim;
  auto kv_compute = AttnMatMul<T>(ctx.cuda_device_context(), false, false, kv_m,
                                  kv_n, kv_k, false);
  kv_compute.ComputeForward(key_weight, key, nullptr, key_out, nullptr);

  // value_out = GEMM(value, value_weight)
  kv_compute.ComputeForward(value_weight, key, nullptr, value_out, nullptr);
}

template <typename T>
Tensor *ComputeSeparatedQKVMatmulBackward(
    const framework::ExecutionContext &ctx,
    const GateAttentionGradConfig<T> &config, const Tensor *query,
    const Tensor *key, const Tensor *query_out_grad, const Tensor *key_out_grad,
    const Tensor *value_out_grad, Tensor *query_grad, Tensor *key_grad,
    bool use_addto) {
  // Gradient of GEMM(key, k_weight)
  const auto *key_weight = ctx.Input<Tensor>("KeyWeight");
  auto *key_weight_grad =
      ctx.Output<Tensor>(framework::GradVarName("KeyWeight"));
  key_weight_grad->mutable_data<T>(ctx.GetPlace());

  int kv_m = config.batch_size * config.seq_len_m * config.m_size;
  int kv_n = config.num_heads * config.key_dim;
  int kv_k = config.kv_dim;
  auto kv_compute = AttnMatMul<T>(ctx.cuda_device_context(), false, false, kv_m,
                                  kv_n, kv_k, false);
  kv_compute.ComputeBackward(key, key_weight, key_out_grad, key_grad,
                             key_weight_grad, nullptr, false);

  // Gradient of GEMM(value, v_weight)
  auto *value_weight = ctx.Input<Tensor>("ValueWeight");
  auto *value_weight_grad =
      ctx.Output<Tensor>(framework::GradVarName("ValueWeight"));
  value_weight_grad->mutable_data<T>(ctx.GetPlace());

  kv_compute.ComputeBackward(key, value_weight, value_out_grad, key_grad,
                             value_weight_grad, nullptr, true);

  // Gradient of GEMM(query, query_weight)
  const auto *query_weight = ctx.Input<Tensor>("QueryWeight");
  auto *query_weight_grad =
      ctx.Output<Tensor>(framework::GradVarName("QueryWeight"));
  query_weight_grad->mutable_data<T>(ctx.GetPlace());

  int q_m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int q_n = config.num_heads * config.key_dim;
  int q_k = config.q_dim;
  auto q_compute = AttnMatMul<T>(ctx.cuda_device_context(), false, false, q_m,
                                 q_n, q_k, false);
  q_compute.ComputeBackward(query, query_weight, query_out_grad, query_grad,
                            query_weight_grad, nullptr, use_addto);
  return query_grad;
}

template <typename T>
Tensor *ComputeGatingLinearForward(const framework::ExecutionContext &ctx,
                                   const GateAttentionConfig<T> &config,
                                   const Tensor *query,
                                   const Tensor *fmha_out) {
  auto *gate_weight = ctx.Input<Tensor>("GateWeight");
  auto *gate_bias = ctx.Input<Tensor>("GateBias");

  auto *gate_out = ctx.Output<Tensor>("GateOut");
  gate_out->mutable_data<T>(ctx.GetPlace());
  VLOG(4) << "[ComputeGatingLinearForward] gate_out: "
          << MemoryDebugString(*gate_out);

  // The first gate_bias_out stores the result of the multiplication,
  // and the second gate_bias_out stores the result of the multiplication +
  // bias.
  //   gate_out = GEMM(query, gate_weight) + gate_bias
  int m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int n = config.num_heads * config.key_dim;
  int k = config.q_dim;
  auto gate_attn_compute =
      AttnMatMul<T>(ctx.cuda_device_context(), false, false, m, n, k, true);
  gate_attn_compute.ComputeForward(gate_weight, query, gate_bias, gate_out,
                                   gate_out);

  // gate_out = sigmoid(gate_out) * fmha_out
  std::vector<const Tensor *> ins = {gate_out, fmha_out};
  std::vector<Tensor *> outs = {gate_out};
  phi::funcs::ElementwiseKernel<T>(ctx.cuda_device_context(), ins, &outs,
                                   SigmoidMultiplyFunctor<T>());
  return gate_out;
}

template <typename T>
Tensor *ComputeGatingLinearBackward(const framework::ExecutionContext &ctx,
                                    const GateAttentionGradConfig<T> &config,
                                    const Tensor *fmha_out,
                                    const Tensor *gate_out_grad,
                                    Tensor *query_grad, Tensor *fmha_out_grad) {
  const auto *query = ctx.Input<Tensor>("Query");
  const auto *gate_weight = ctx.Input<Tensor>("GateWeight");
  const auto *gate_bias = ctx.Input<Tensor>("GateBias");

  // Re-compute gate_bias_out
  Tensor gate_bias_out;
  gate_bias_out.Resize(config.gate_out_dims);
  gate_bias_out.mutable_data<T>(ctx.GetPlace());

  int m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int n = config.num_heads * config.key_dim;
  int k = config.q_dim;
  auto gate_attn_compute =
      AttnMatMul<T>(ctx.cuda_device_context(), false, false, m, n, k, true);
  gate_attn_compute.ComputeForward(gate_weight, query, gate_bias,
                                   &gate_bias_out, &gate_bias_out);

  // Gradient of sigmoid(gate_bias_out) * fmha_out
  // Compute inplace and save gate_bias_out_grad to gate_bias_out.
  std::vector<const Tensor *> ins = {gate_out_grad, &gate_bias_out, fmha_out};
  std::vector<Tensor *> outs = {&gate_bias_out, fmha_out_grad};
  phi::funcs::ElementwiseKernel<T, SigmoidMultiplyGradFunctor<T>, 2>(
      ctx.cuda_device_context(), ins, &outs, SigmoidMultiplyGradFunctor<T>());

  // Gradient of GEMM(query, gate_weight) + gate_bias
  auto *gate_weight_grad =
      ctx.Output<Tensor>(framework::GradVarName("GateWeight"));
  auto *gate_bias_grad = ctx.Output<Tensor>(framework::GradVarName("GateBias"));
  gate_weight_grad->mutable_data<T>(ctx.GetPlace());
  gate_bias_grad->mutable_data<T>(ctx.GetPlace());

  gate_attn_compute.ComputeBackward(query, gate_weight, &gate_bias_out,
                                    query_grad, gate_weight_grad,
                                    gate_bias_grad);
  return fmha_out_grad;
}

template <typename T>
Tensor *ComputeOutputLinearForward(const framework::ExecutionContext &ctx,
                                   const GateAttentionConfig<T> &config,
                                   const Tensor *fmha_or_gate_out) {
  const auto *out_linear_weight = ctx.Input<Tensor>("OutLinearWeight");
  const auto *out_linear_bias = ctx.Input<Tensor>("OutLinearBias");

  auto *out = ctx.Output<Tensor>("Out");
  out->mutable_data<T>(ctx.GetPlace());
  VLOG(4) << "[ComputeOutputLinearForward] out: " << MemoryDebugString(*out);

  // out = GEMM(fmha_or_gate_out, out_linear_weight) + out_linear_bias
  int m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int n = config.q_dim;
  int k = config.num_heads * config.key_dim;
  auto out_linear_compute =
      AttnMatMul<T>(ctx.cuda_device_context(), false, false, m, n, k, true);
  out_linear_compute.ComputeForward(out_linear_weight, fmha_or_gate_out,
                                    out_linear_bias, out, out);
  return out;
}

template <typename T>
Tensor *ComputeOutputLinearBackward(const framework::ExecutionContext &ctx,
                                    const GateAttentionGradConfig<T> &config,
                                    bool has_gating) {
  std::string input_name = has_gating ? "GateOut" : "FMHAOut";

  const auto *out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
  const auto *out_linear_weight = ctx.Input<Tensor>("OutLinearWeight");
  const auto *input = ctx.Input<Tensor>(input_name);

  auto *out_linear_weight_grad =
      ctx.Output<Tensor>(framework::GradVarName("OutLinearWeight"));
  auto *out_linear_bias_grad =
      ctx.Output<Tensor>(framework::GradVarName("OutLinearBias"));
  auto *input_grad = ctx.Output<Tensor>(framework::GradVarName(input_name));

  out_linear_weight_grad->mutable_data<T>(ctx.GetPlace());
  out_linear_bias_grad->mutable_data<T>(ctx.GetPlace());
  input_grad->mutable_data<T>(ctx.GetPlace());

  int m = config.batch_size * config.seq_len_m * config.seq_len_r;
  int n = config.q_dim;
  int k = config.num_heads * config.key_dim;
  auto out_linear_compute =
      AttnMatMul<T>(ctx.cuda_device_context(), false, false, m, n, k, true);
  out_linear_compute.ComputeBackward(input, out_linear_weight, out_grad,
                                     input_grad, out_linear_weight_grad,
                                     out_linear_bias_grad);
  return input_grad;
}

template <typename T>
class FusedGateAttentionOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *query = ctx.Input<Tensor>("Query");
    const auto *key = ctx.Input<Tensor>("Key");
    const auto *query_weight = ctx.Input<Tensor>("QueryWeight");
    const auto *qkv_weight = ctx.Input<Tensor>("QKVWeight");

    const auto *src_mask = ctx.Input<Tensor>("SrcMask");
    const auto *nonbatched_bias = ctx.Input<Tensor>("NonbatchedBias");

    auto *q_transpose_out = ctx.Output<Tensor>("QueryTransposeOut");
    auto *k_transpose_out = ctx.Output<Tensor>("KeyTransposeOut");
    auto *v_transpose_out = ctx.Output<Tensor>("ValueTransposeOut");
    auto *qkv_transpose_out = ctx.Output<Tensor>("QKVTransposeOut");

    auto *softmax_out = ctx.Output<Tensor>("SoftmaxOut");
    auto *fmha_out = ctx.Output<Tensor>("FMHAOut");

    const bool merge_qkv = ctx.Attr<bool>("merge_qkv");
    const bool has_gating = ctx.Attr<bool>("has_gating");

    // When seq_len_r = m_size, q_dim = kv_dim, QKV matmul can be merged.
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    GateAttentionConfig<T> config(query, key, query_weight, qkv_weight,
                                  merge_qkv);

    if (merge_qkv) {
      // 1. Merged QKV Matmul: einsum(nbhqk,nbkhc -> nbqhc)
      Tensor *qkv_out = config.GetQKVOut(dev_ctx);
      ComputeMergedQKVMatmulForward<T>(ctx, config, query, qkv_out);

      qkv_transpose_out->mutable_data<T>(ctx.GetPlace());
      VLOG(4) << "qkv_transpose_out:" << MemoryDebugString(*qkv_transpose_out);
    } else {
      // 1. Separated QKV Matmul
      Tensor *query_out = config.GetQueryOut(dev_ctx);
      Tensor *key_out = config.GetKeyOut(dev_ctx);
      Tensor *value_out = config.GetValueOut(dev_ctx);
      ComputeSeparatedQKVMatmulForward<T>(ctx, config, query, key, query_out,
                                          key_out, value_out);

      q_transpose_out->mutable_data<T>(ctx.GetPlace());
      k_transpose_out->mutable_data<T>(ctx.GetPlace());
      v_transpose_out->mutable_data<T>(ctx.GetPlace());
      VLOG(4) << "q_transpose_out: " << MemoryDebugString(*q_transpose_out);
      VLOG(4) << "k_transpose_out: " << MemoryDebugString(*k_transpose_out);
      VLOG(4) << "v_transpose_out: " << MemoryDebugString(*v_transpose_out);
    }

    softmax_out->mutable_data<T>(ctx.GetPlace());
    fmha_out->mutable_data<T>(ctx.GetPlace());
    VLOG(4) << "softmax_out: " << MemoryDebugString(*softmax_out);
    VLOG(4) << "fmha_out: " << MemoryDebugString(*fmha_out);

    // 2. FMHA
    auto fmha_compute = FMHAGateRef<T>(dev_ctx, merge_qkv);
    fmha_compute.ComputeForward(
        nonbatched_bias, src_mask, q_transpose_out, k_transpose_out,
        v_transpose_out, qkv_transpose_out, softmax_out, fmha_out, &config);

    // 3. Gating Linear
    Tensor *fmha_or_gate_out =
        !has_gating ? fmha_out : ComputeGatingLinearForward<T>(ctx, config,
                                                               query, fmha_out);

    // 4. Output Linear
    ComputeOutputLinearForward<T>(ctx, config, fmha_or_gate_out);
  }
};

template <typename T>
class FusedGateAttentionGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto has_gating = ctx.Attr<bool>("has_gating");
    const auto merge_qkv = ctx.Attr<bool>("merge_qkv");

    // forward input
    const auto *query = ctx.Input<Tensor>("Query");
    const auto *key = ctx.Input<Tensor>("Key");
    const auto *query_weight = ctx.Input<Tensor>("QueryWeight");
    const auto *qkv_weight = ctx.Input<Tensor>("QKVWeight");

    // forward output, backward input
    const auto *q_transpose_out = ctx.Input<Tensor>("QueryTransposeOut");
    const auto *k_transpose_out = ctx.Input<Tensor>("KeyTransposeOut");
    const auto *v_transpose_out = ctx.Input<Tensor>("ValueTransposeOut");
    const auto *qkv_transpose_out = ctx.Input<Tensor>("QKVTransposeOut");
    const auto *softmax_out = ctx.Input<Tensor>("SoftmaxOut");
    const auto *fmha_out = ctx.Input<Tensor>("FMHAOut");

    // backward output
    auto *query_grad = ctx.Output<Tensor>(framework::GradVarName("Query"));
    query_grad->mutable_data<T>(ctx.GetPlace());
    auto *q_transpose_out_grad =
        ctx.Output<Tensor>(framework::GradVarName("QueryTransposeOut"));
    auto *k_transpose_out_grad =
        ctx.Output<Tensor>(framework::GradVarName("KeyTransposeOut"));
    auto *v_transpose_out_grad =
        ctx.Output<Tensor>(framework::GradVarName("ValueTransposeOut"));
    auto *qkv_transpose_out_grad =
        ctx.Output<Tensor>(framework::GradVarName("QKVTransposeOut"));
    auto *nonbatched_bias_grad =
        ctx.Output<Tensor>(framework::GradVarName("NonbatchedBias"));
    auto *softmax_out_grad =
        ctx.Output<Tensor>(framework::GradVarName("SoftmaxOut"));
    auto *fmha_out_grad = ctx.Output<Tensor>(framework::GradVarName("FMHAOut"));

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    GateAttentionGradConfig<T> config(query, key, query_weight, qkv_weight,
                                      merge_qkv);

    // 1. Gradient of Output Linear
    Tensor *fhma_or_gate_out_grad =
        ComputeOutputLinearBackward<T>(ctx, config, has_gating);

    // 2. Gradient of Gating Linear
    if (has_gating) {
      // fhma_or_gate_out_grad is actually gate_out_grad.
      fmha_out_grad->mutable_data<T>(ctx.GetPlace());
      ComputeGatingLinearBackward<T>(ctx, config, fmha_out,
                                     fhma_or_gate_out_grad, query_grad,
                                     fmha_out_grad);
    }

    // 3. Gradient of FMHA
    if (merge_qkv) {
      qkv_transpose_out_grad->mutable_data<T>(ctx.GetPlace());
    } else {
      q_transpose_out_grad->mutable_data<T>(ctx.GetPlace());
      k_transpose_out_grad->mutable_data<T>(ctx.GetPlace());
      v_transpose_out_grad->mutable_data<T>(ctx.GetPlace());
    }
    if (nonbatched_bias_grad) {
      nonbatched_bias_grad->mutable_data<T>(ctx.GetPlace());
    }
    softmax_out_grad->mutable_data<T>(ctx.GetPlace());

    auto fmha_compute = FMHAGateRef<T>(dev_ctx, merge_qkv);
    fmha_compute.ComputeBackward(
        q_transpose_out, k_transpose_out, v_transpose_out, qkv_transpose_out,
        softmax_out, fmha_out_grad, softmax_out_grad, nullptr,
        nonbatched_bias_grad, q_transpose_out_grad, k_transpose_out_grad,
        v_transpose_out_grad, qkv_transpose_out_grad, &config);

    bool use_addto = has_gating ? true : false;
    if (merge_qkv) {
      // 4. Gradient of Merged QKV Matmul
      Tensor *qkv_out_grad = config.GetQKVOutGrad(dev_ctx);
      ComputeMergedQKVMatmulBackward<T>(ctx, config, query, qkv_out_grad,
                                        query_grad, use_addto);
    } else {
      // 4. Gradient of Separated QKV Matmul
      auto *key_grad = ctx.Output<Tensor>(framework::GradVarName("Key"));
      if (key_grad) {
        key_grad->mutable_data<T>(ctx.GetPlace());
      }
      Tensor *query_out_grad = config.GetQueryOutGrad(dev_ctx);
      Tensor *key_out_grad = config.GetKeyOutGrad(dev_ctx);
      Tensor *value_out_grad = config.GetValueOutGrad(dev_ctx);
      ComputeSeparatedQKVMatmulBackward<T>(
          ctx, config, query, key, query_out_grad, key_out_grad, value_out_grad,
          query_grad, key_grad, use_addto);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
#ifdef PADDLE_WITH_HIP
REGISTER_OP_CUDA_KERNEL(fused_gate_attention,
                        ops::FusedGateAttentionOpKernel<float>,
                        ops::FusedGateAttentionOpKernel<plat::float16>,
                        ops::FusedGateAttentionOpKernel<plat::bfloat16>);
REGISTER_OP_CUDA_KERNEL(fused_gate_attention_grad,
                        ops::FusedGateAttentionGradKernel<float>,
                        ops::FusedGateAttentionGradKernel<plat::float16>,
                        ops::FusedGateAttentionGradKernel<plat::bfloat16>);
#else
REGISTER_OP_CUDA_KERNEL(fused_gate_attention,
                        ops::FusedGateAttentionOpKernel<float>,
                        ops::FusedGateAttentionOpKernel<double>,
                        ops::FusedGateAttentionOpKernel<plat::float16>,
                        ops::FusedGateAttentionOpKernel<plat::bfloat16>);
REGISTER_OP_CUDA_KERNEL(fused_gate_attention_grad,
                        ops::FusedGateAttentionGradKernel<float>,
                        ops::FusedGateAttentionGradKernel<double>,
                        ops::FusedGateAttentionGradKernel<plat::float16>,
                        ops::FusedGateAttentionGradKernel<plat::bfloat16>);
#endif
