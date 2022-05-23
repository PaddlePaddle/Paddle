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

#ifdef PADDLE_WITH_CUDA
#include <cuda_fp16.h>
#endif
#ifdef PADDLE_WITH_HIP
#include <hip/hip_fp16.h>
#endif
#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/fused/attn_gemm.h"
#include "paddle/fluid/operators/fused/fused_gate_attention.h"
#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DDim = framework::DDim;

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
                                   const Tensor *x, Tensor *qkv_out, int m,
                                   int n, int k) {
  auto *qkv_weight = ctx.Input<Tensor>("QKVWeight");

  // qkv_out = GEMM(x, qkv_weight)
  auto qkv_compute =
      AttnMatMul<T>(ctx.cuda_device_context(), false, true, m, n, k, false);
  qkv_compute.ComputeForward(qkv_weight, x, nullptr, qkv_out, nullptr);
}

template <typename T>
Tensor *ComputeMergedQKVMatmulBackward(const framework::ExecutionContext &ctx,
                                       const Tensor *x, const Tensor *d_qkv_out,
                                       Tensor *d_x, int m, int n, int k,
                                       bool use_addto) {
  auto *qkv_weight = ctx.Input<Tensor>("QKVWeight");
  auto *d_qkv_weight = ctx.Output<Tensor>(framework::GradVarName("QKVWeight"));
  d_qkv_weight->mutable_data<T>(ctx.GetPlace());

  // Gradient of GEMM(x, qkv_weight)
  auto qkv_compute =
      AttnMatMul<T>(ctx.cuda_device_context(), false, true, m, n, k, false);
  qkv_compute.ComputeBackward(x, qkv_weight, d_qkv_out, d_x, d_qkv_weight,
                              nullptr, use_addto);
  return d_x;
}

template <typename T>
void ComputeQKVMatmulForward(const framework::ExecutionContext &ctx,
                             const Tensor *query, const Tensor *key,
                             Tensor *q_out, Tensor *k_out, Tensor *v_out,
                             int q_m, int q_n, int q_k, int k_m, int k_n,
                             int k_k) {
  auto *q_weight = ctx.Input<Tensor>("QueryWeight");
  auto *k_weight = ctx.Input<Tensor>("KeyWeight");
  auto *v_weight = ctx.Input<Tensor>("ValueWeight");

  // q_out = GEMM(query, q_weight)
  auto q_compute = AttnMatMul<T>(ctx.cuda_device_context(), false, false, q_m,
                                 q_n, q_k, false);
  q_compute.ComputeForward(q_weight, query, nullptr, q_out, nullptr);

  // k_out = GEMM(key, k_weight)
  auto k_compute = AttnMatMul<T>(ctx.cuda_device_context(), false, false, k_m,
                                 k_n, k_k, false);
  k_compute.ComputeForward(k_weight, key, nullptr, k_out, nullptr);

  // v_out = GEMM(value, v_weight)
  k_compute.ComputeForward(v_weight, key, nullptr, v_out, nullptr);
}

template <typename T>
Tensor *ComputeQKVMatmulBackward(const framework::ExecutionContext &ctx,
                                 const Tensor *query, const Tensor *key,
                                 const Tensor *d_q_out, const Tensor *d_k_out,
                                 const Tensor *d_v_out, Tensor *d_x,
                                 Tensor *d_key, int q_m, int q_n, int q_k,
                                 int k_m, int k_n, int k_k, bool use_addto) {
  auto *q_weight = ctx.Input<Tensor>("QueryWeight");
  auto *d_q_weight = ctx.Output<Tensor>(framework::GradVarName("QueryWeight"));
  auto *k_weight = ctx.Input<Tensor>("KeyWeight");
  auto *d_k_weight = ctx.Output<Tensor>(framework::GradVarName("KeyWeight"));
  auto *v_weight = ctx.Input<Tensor>("ValueWeight");
  auto *d_v_weight = ctx.Output<Tensor>(framework::GradVarName("ValueWeight"));
  d_q_weight->mutable_data<T>(ctx.GetPlace());
  d_k_weight->mutable_data<T>(ctx.GetPlace());
  d_v_weight->mutable_data<T>(ctx.GetPlace());

  // Gradient of GEMM(key, k_weight)
  auto k_compute = AttnMatMul<T>(ctx.cuda_device_context(), false, false, k_m,
                                 k_n, k_k, false);
  k_compute.ComputeBackward(key, k_weight, d_k_out, d_key, d_k_weight, nullptr,
                            use_addto);

  // Gradient of GEMM(value, v_weight)
  k_compute.ComputeBackward(key, v_weight, d_v_out, d_key, d_v_weight, nullptr,
                            use_addto);

  // Gradient of GEMM(query, q_weight)
  auto q_compute = AttnMatMul<T>(ctx.cuda_device_context(), false, false, q_m,
                                 q_n, q_k, false);
  q_compute.ComputeBackward(query, q_weight, d_q_out, d_x, d_q_weight, nullptr,
                            use_addto);
  return d_x;
}

template <typename T>
Tensor *ComputeGatingLinearForward(const framework::ExecutionContext &ctx,
                                   const Tensor *x, const Tensor *fmha_out,
                                   int m, int n, int k) {
  auto *gate_weight = ctx.Input<Tensor>("GateWeight");
  auto *gate_bias = ctx.Input<Tensor>("GateBias");

  auto *gate_out = ctx.Output<Tensor>("GateOut");
  gate_out->mutable_data<T>(ctx.GetPlace());
  VLOG(4) << "[ComputeGatingLinearForward] gate_out: "
          << MemoryDebugString(*gate_out);

  // The first gate_bias_out stores the result of the multiplication,
  // and the second gate_bias_out stores the result of the multiplication +
  // bias.
  //   gate_out = GEMM(x, gate_weight)
  //   gate_out = gate_out + gate_bias
  auto gate_attn_compute =
      AttnMatMul<T>(ctx.cuda_device_context(), false, false, m, n, k, true);
  gate_attn_compute.ComputeForward(gate_weight, x, gate_bias, gate_out,
                                   gate_out);

  // gate_out = sigmoid(gate_out) * fmha_out
  std::vector<const Tensor *> ins = {gate_out, fmha_out};
  std::vector<Tensor *> outs = {gate_out};
  paddle::operators::LaunchSameDimsElementwiseCudaKernel<T>(
      ctx.cuda_device_context(), ins, &outs, SigmoidMultiplyFunctor<T>());
  return gate_out;
}

template <typename T>
Tensor *ComputeGatingLinearBackward(const framework::ExecutionContext &ctx,
                                    const Tensor *fmha_out,
                                    const Tensor *d_gate_out, Tensor *d_x,
                                    Tensor *d_fmha_out, Tensor *d_gate_bias_out,
                                    Tensor *gate_bias_out, int m, int n,
                                    int k) {
  auto *query = ctx.Input<Tensor>("Query");
  auto *gate_weight = ctx.Input<Tensor>("GateWeight");
  auto *gate_bias = ctx.Input<Tensor>("GateBias");

  auto gate_attn_compute =
      AttnMatMul<T>(ctx.cuda_device_context(), false, false, m, n, k, true);
  gate_attn_compute.ComputeForward(gate_weight, query, gate_bias, gate_bias_out,
                                   gate_bias_out);

  auto *d_gate_weight =
      ctx.Output<Tensor>(framework::GradVarName("GateWeight"));
  auto *d_gate_bias = ctx.Output<Tensor>(framework::GradVarName("GateBias"));

  d_gate_weight->mutable_data<T>(ctx.GetPlace());
  d_gate_bias->mutable_data<T>(ctx.GetPlace());

  // Gradient of sigmoid(gate_bias_out) * fmha_out
  std::vector<const Tensor *> ins = {d_gate_out, gate_bias_out, fmha_out};
  std::vector<Tensor *> outs = {d_gate_bias_out, d_fmha_out};
  paddle::operators::LaunchSameDimsElementwiseCudaKernel<
      T, SigmoidMultiplyGradFunctor<T>, 2>(
      ctx.cuda_device_context(), ins, &outs, SigmoidMultiplyGradFunctor<T>());

  gate_attn_compute.ComputeBackward(query, gate_weight, d_gate_bias_out, d_x,
                                    d_gate_weight, d_gate_bias);
  return d_fmha_out;
}

template <typename T>
Tensor *ComputeOutputLinearForward(const framework::ExecutionContext &ctx,
                                   const Tensor *fmha_or_gate_out, int m, int n,
                                   int k) {
  auto *out_linear_weight = ctx.Input<Tensor>("OutLinearWeight");
  auto *out_linear_bias = ctx.Input<Tensor>("OutLinearBias");

  auto *out = ctx.Output<Tensor>("Out");
  out->mutable_data<T>(ctx.GetPlace());
  VLOG(4) << "[ComputeOutputLinearForward] out: " << MemoryDebugString(*out);

  // out = GEMM(gate_out, out_linear_weight)
  // out = out + out_linear_bias
  auto out_linear_compute =
      AttnMatMul<T>(ctx.cuda_device_context(), false, false, m, n, k, true);
  out_linear_compute.ComputeForward(out_linear_weight, fmha_or_gate_out,
                                    out_linear_bias, out, out);
  return out;
}

template <typename T>
Tensor *ComputeOutputLinearBackward(const framework::ExecutionContext &ctx,
                                    const Tensor *fmha_out, Tensor *d_fmha_out,
                                    int m, int n, int k, bool has_gating) {
  auto *d_out = ctx.Input<Tensor>(framework::GradVarName("Out"));
  auto *out_linear_weight = ctx.Input<Tensor>("OutLinearWeight");
  auto *input = has_gating ? ctx.Input<Tensor>("GateOut") : fmha_out;

  auto *d_out_linear_weight =
      ctx.Output<Tensor>(framework::GradVarName("OutLinearWeight"));
  auto *d_out_linear_bias =
      ctx.Output<Tensor>(framework::GradVarName("OutLinearBias"));
  auto *d_input = has_gating
                      ? ctx.Output<Tensor>(framework::GradVarName("GateOut"))
                      : d_fmha_out;

  d_out_linear_weight->mutable_data<T>(ctx.GetPlace());
  d_out_linear_bias->mutable_data<T>(ctx.GetPlace());
  d_input->mutable_data<T>(ctx.GetPlace());

  auto out_linear_compute =
      AttnMatMul<T>(ctx.cuda_device_context(), false, false, m, n, k, true);
  out_linear_compute.ComputeBackward(input, out_linear_weight, d_out, d_input,
                                     d_out_linear_weight, d_out_linear_bias);
  return d_input;
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

    auto *query_out = ctx.Output<Tensor>("QueryOut");
    auto *key_out = ctx.Output<Tensor>("KeyOut");
    auto *value_out = ctx.Output<Tensor>("ValueOut");
    auto *qkv_out = ctx.Output<Tensor>("QKVOut");

    auto *softmax_out = ctx.Output<Tensor>("SoftmaxOut");
    auto *qktv_out = ctx.Output<Tensor>("QKTVOut");

    const bool merge_qkv = ctx.Attr<bool>("merge_qkv");
    const bool has_gating = ctx.Attr<bool>("has_gating");

    // When seq_len_r = m_size, q_dim = kv_dim, QKV matmul can be merged.
    GateAttentionConfig<T> config(query, key, query_weight, qkv_weight,
                                  merge_qkv);

    if (merge_qkv) {
      // 1. Merged QKV Matmul: einsum(nbhqk,nbkhc -> nbqhc)
      //    query: shape=[batch_size, seq_len_m, seq_len_r, qkv_dim]
      //    qkv_weight: shape=[3, num_heads, key_dim, qkv_dim]
      //    qkv_out: shape=[batch_size, seq_len_m, seq_len_r, 3, num_heads,
      //    key_dim]
      qkv_out->mutable_data<T>(ctx.GetPlace());
      VLOG(4) << "qkv_out:" << MemoryDebugString(*qkv_out);

      int m = config.batch_size * config.seq_len_m * config.seq_len_r;
      int n = 3 * config.num_heads * config.key_dim;
      int k = config.q_dim;

      ComputeMergedQKVMatmulForward<T>(ctx, query, qkv_out, m, n, k);
    } else {
      // 1. Separated QKV Matmul
      query_out->mutable_data<T>(ctx.GetPlace());
      key_out->mutable_data<T>(ctx.GetPlace());
      value_out->mutable_data<T>(ctx.GetPlace());
      VLOG(4) << "query_out: " << MemoryDebugString(*query_out);
      VLOG(4) << "key_out: " << MemoryDebugString(*key_out);
      VLOG(4) << "value_out: " << MemoryDebugString(*value_out);

      int q_m = config.batch_size * config.seq_len_m * config.seq_len_r;
      int q_n = config.num_heads * config.key_dim;
      int q_k = config.q_dim;

      int k_m = config.batch_size * config.seq_len_m * config.m_size;
      int k_n = config.num_heads * config.key_dim;
      int k_k = config.kv_dim;

      ComputeQKVMatmulForward<T>(ctx, query, key, query_out, key_out, value_out,
                                 q_m, q_n, q_k, k_m, k_n, k_k);
    }

    softmax_out->mutable_data<T>(ctx.GetPlace());
    qktv_out->mutable_data<T>(ctx.GetPlace());
    VLOG(4) << "softmax_out: " << MemoryDebugString(*softmax_out);
    VLOG(4) << "qktv_out: " << MemoryDebugString(*qktv_out);

    // 2. FMHA
    auto fmha_compute =
        FMHAGateRef<T>(ctx.cuda_device_context(), merge_qkv, &config);

    Tensor *fmha_out = config.GetFMHAOut(ctx.cuda_device_context());
    fmha_compute.ComputeForward(query_out, key_out, value_out, qkv_out,
                                nonbatched_bias, src_mask, softmax_out,
                                qktv_out, fmha_out);

    // 3. Gating Linear
    Tensor *fmha_or_gate_out = fmha_out;
    if (has_gating) {
      int m = config.batch_size * config.seq_len_m * config.seq_len_r;
      int n = config.num_heads * config.key_dim;
      int k = config.q_dim;
      fmha_or_gate_out =
          ComputeGatingLinearForward<T>(ctx, query, fmha_out, m, n, k);
    }

    // 4. Output Linear
    int m = config.batch_size * config.seq_len_m * config.seq_len_r;
    int n = config.q_dim;
    int k = config.num_heads * config.key_dim;
    ComputeOutputLinearForward<T>(ctx, fmha_or_gate_out, m, n, k);
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

    auto *nonbatched_bias = ctx.Input<Tensor>("NonbatchedBias");

    // forward output, backward input
    auto *softmax_out = ctx.Input<Tensor>("SoftmaxOut");
    auto *qktv_out = ctx.Input<Tensor>("QKTVOut");
    auto *q_out = ctx.Input<Tensor>("QueryOut");
    auto *key_out = ctx.Input<Tensor>("KeyOut");
    auto *value_out = ctx.Input<Tensor>("ValueOut");
    auto *qkv_out = ctx.Input<Tensor>("QKVOut");

    // backward output
    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("Query"));
    auto *d_key = ctx.Output<Tensor>(framework::GradVarName("Key"));
    auto *d_qktv_out = ctx.Output<Tensor>(framework::GradVarName("QKTVOut"));
    auto *d_softmax_out =
        ctx.Output<Tensor>(framework::GradVarName("SoftmaxOut"));
    auto *d_qkv_out = ctx.Output<Tensor>(framework::GradVarName("QKVOut"));
    auto *d_q_out = ctx.Output<Tensor>(framework::GradVarName("QueryOut"));
    auto *d_k_out = ctx.Output<Tensor>(framework::GradVarName("KeyOut"));
    auto *d_v_out = ctx.Output<Tensor>(framework::GradVarName("ValueOut"));

    d_x->mutable_data<T>(ctx.GetPlace());
    d_qktv_out->mutable_data<T>(ctx.GetPlace());
    d_softmax_out->mutable_data<T>(ctx.GetPlace());

    // parameter grad
    auto *d_nonbatched_bias =
        ctx.Output<Tensor>(framework::GradVarName("NonbatchedBias"));

    if (nonbatched_bias) {
      d_nonbatched_bias->mutable_data<T>(ctx.GetPlace());
    }

    if (key) {
      d_key->mutable_data<T>(ctx.GetPlace());
    }

    GateAttentionConfig<T> config(query, key, query_weight, qkv_weight,
                                  merge_qkv);

    if (merge_qkv) {
      // qkv_weight[3, n_head, c, qkv_dim]
      d_qkv_out->mutable_data<T>(ctx.GetPlace());
    } else {
      d_q_out->mutable_data<T>(ctx.GetPlace());
      d_k_out->mutable_data<T>(ctx.GetPlace());
      d_v_out->mutable_data<T>(ctx.GetPlace());
    }

    auto fmha_compute =
        FMHAGateRef<T>(ctx.cuda_device_context(), merge_qkv, &config);

    // Re-compute fmha_out.
    Tensor *fmha_out = config.GetFMHAOut(ctx.cuda_device_context());
    fmha_compute.ComputeQKTVTransposeForward(*qktv_out, fmha_out);

    // 1. Gradient of Output Linear
    int m = config.batch_size * config.seq_len_m * config.seq_len_r;
    int n = config.q_dim;
    int k = config.num_heads * config.key_dim;

    Tensor *d_fmha_out = config.GetFMHAOutGrad(ctx.cuda_device_context());
    Tensor *d_fhma_or_gate_out = ComputeOutputLinearBackward<T>(
        ctx, fmha_out, d_fmha_out, m, n, k, has_gating);

    // 2. Gradient of Gating Linear
    if (has_gating) {
      m = config.batch_size * config.seq_len_m * config.seq_len_r;
      n = config.num_heads * config.key_dim;
      k = config.q_dim;

      Tensor gate_bias_out;
      gate_bias_out.Resize(config.gate_out_dims);
      gate_bias_out.mutable_data<T>(ctx.GetPlace());

      Tensor d_gate_bias_out;
      d_gate_bias_out.Resize(config.gate_out_dims);
      d_gate_bias_out.mutable_data<T>(ctx.GetPlace());

      // d_fhma_or_gate_out is d_gate_out.
      ComputeGatingLinearBackward<T>(ctx, fmha_out, d_fhma_or_gate_out, d_x,
                                     d_fmha_out, &d_gate_bias_out,
                                     &gate_bias_out, m, n, k);
    }

    // Re-compute the qkv_transpose_out.
    Tensor qkv_transpose_out, d_qkv_transpose_out;
    Tensor q_transpose_out, k_transpose_out, v_transpose_out, d_q_transpose_out,
        d_k_transpose_out, d_v_transpose_out;
    if (merge_qkv) {
      qkv_transpose_out.Resize(config.qkv_transpose_out_dims);
      qkv_transpose_out.mutable_data<T>(ctx.GetPlace());
      fmha_compute.ComputeQKVTransposeForward(*qkv_out, &qkv_transpose_out);

      d_qkv_transpose_out.Resize(config.qkv_transpose_out_dims);
      d_qkv_transpose_out.mutable_data<T>(ctx.GetPlace());
    } else {
      q_transpose_out.Resize(config.q_transpose_out_dims);
      q_transpose_out.mutable_data<T>(ctx.GetPlace());

      k_transpose_out.Resize(config.kv_transpose_out_dims);
      k_transpose_out.mutable_data<T>(ctx.GetPlace());

      v_transpose_out.Resize(config.kv_transpose_out_dims);
      v_transpose_out.mutable_data<T>(ctx.GetPlace());
      fmha_compute.ComputeQKVTransposeForward(
          *q_out, *key_out, *value_out, &q_transpose_out, &k_transpose_out,
          &v_transpose_out);

      d_q_transpose_out.Resize(config.q_transpose_out_dims);
      d_q_transpose_out.mutable_data<T>(ctx.GetPlace());

      d_k_transpose_out.Resize(config.kv_transpose_out_dims);
      d_k_transpose_out.mutable_data<T>(ctx.GetPlace());

      d_v_transpose_out.Resize(config.kv_transpose_out_dims);
      d_v_transpose_out.mutable_data<T>(ctx.GetPlace());
    }

    // 3. Gradient of FMHA
    fmha_compute.ComputeBackward(
        q_transpose_out, k_transpose_out, v_transpose_out, qkv_transpose_out,
        *softmax_out, *d_fmha_out, nonbatched_bias, d_nonbatched_bias,
        d_qktv_out, d_softmax_out, nullptr, &d_q_transpose_out,
        &d_k_transpose_out, &d_v_transpose_out, d_q_out, d_k_out, d_v_out,
        &d_qkv_transpose_out, d_qkv_out);

    bool use_addto = has_gating ? true : false;
    if (merge_qkv) {
      // 4. Gradient of Merged QKV Matmul
      m = config.batch_size * config.seq_len_m * config.seq_len_r;
      n = 3 * config.num_heads * config.key_dim;
      k = config.q_dim;
      ComputeMergedQKVMatmulBackward<T>(ctx, query, d_qkv_out, d_x, m, n, k,
                                        use_addto);
      if (key) {
        d_key = d_x;
      }
    } else {
      // 4. Gradient of Separated QKV Matmul
      int q_m = config.batch_size * config.seq_len_m * config.seq_len_r;
      int q_n = config.num_heads * config.key_dim;
      int q_k = config.q_dim;

      int k_m = config.batch_size * config.seq_len_m * config.m_size;
      int k_n = config.num_heads * config.key_dim;
      int k_k = config.kv_dim;
      ComputeQKVMatmulBackward<T>(ctx, query, key, d_q_out, d_k_out, d_v_out,
                                  d_x, d_key, q_m, q_n, q_k, k_m, k_n, k_k,
                                  use_addto);
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
