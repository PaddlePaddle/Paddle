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

#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"

#include "paddle/fluid/operators/elementwise/elementwise_add_op.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#include "paddle/fluid/operators/fused/attention_layer_norm.h"
#include "paddle/fluid/operators/fused/attn_gemm.h"
#include "paddle/fluid/operators/fused/fmha_ref.h"
#include "paddle/fluid/operators/fused/fused_dropout_helper.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif
#include "paddle/fluid/memory/memory.h"

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
Tensor *ComputeMergedQKVMatmulForward(const framework::ExecutionContext &ctx,
                                      const Tensor *x, int m, int n, int k) {
  auto *qkv_weight = ctx.Input<Tensor>("QKVWeight");
  auto *qkv_out = ctx.Output<Tensor>("QKVOut");
  qkv_out->mutable_data<T>(ctx.GetPlace());

  // qkv_out = GEMM(x, qkv_weight)
  auto qkv_compute =
      AttnMatMul<T>(ctx.cuda_device_context(), false, true, m, n, k, false);
  qkv_compute.ComputeForward(qkv_weight, x, nullptr, qkv_out, nullptr);
  return qkv_out;
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
Tensor *ComputeGatingLinearForward(const framework::ExecutionContext &ctx,
                                   const Tensor *x, const Tensor *fmha_out,
                                   int m, int n, int k) {
  auto *gate_weight = ctx.Input<Tensor>("GateWeight");
  auto *gate_bias = ctx.Input<Tensor>("GateBias");

  auto *gate_bias_out = ctx.Output<Tensor>("GateBiasOut");
  auto *gate_out = ctx.Output<Tensor>("GateOut");

  gate_bias_out->mutable_data<T>(ctx.GetPlace());
  gate_out->mutable_data<T>(ctx.GetPlace());

  // The first gate_bias_out stores the result of the multiplication,
  // and the second gate_bias_out stores the result of the multiplication +
  // bias.
  //   gate_bias_out = GEMM(x, gate_weight)
  //   gate_bias_out = gate_bias_out + gate_bias
  auto gate_attn_compute =
      AttnMatMul<T>(ctx.cuda_device_context(), false, false, m, n, k, true);
  gate_attn_compute.ComputeForward(gate_weight, x, gate_bias, gate_bias_out,
                                   gate_bias_out);

  std::vector<const Tensor *> ins = {gate_bias_out, fmha_out};
  std::vector<Tensor *> outs = {gate_out};
  paddle::operators::LaunchSameDimsElementwiseCudaKernel<T>(
      ctx.cuda_device_context(), ins, &outs, SigmoidMultiplyFunctor<T>());
  return gate_out;
}

template <typename T>
Tensor *ComputeGatingLinearBackward(const framework::ExecutionContext &ctx,
                                    const Tensor *fmha_out,
                                    const Tensor *d_gate_out, Tensor *d_x,
                                    Tensor *d_fmha_out, int m, int n, int k) {
  auto *gate_bias_out = ctx.Input<Tensor>("GateBiasOut");
  auto *x = ctx.Input<Tensor>("X");
  auto *gate_weight = ctx.Input<Tensor>("GateWeight");

  auto *d_gate_bias_out =
      ctx.Output<Tensor>(framework::GradVarName("GateBiasOut"));
  d_gate_bias_out->mutable_data<T>(ctx.GetPlace());

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

  auto gate_attn_compute =
      AttnMatMul<T>(ctx.cuda_device_context(), false, false, m, n, k, true);
  gate_attn_compute.ComputeBackward(x, gate_weight, d_gate_bias_out, d_x,
                                    d_gate_weight, d_gate_bias);
  return d_fmha_out;
}

template <typename T>
Tensor *ComputeOutputLinearForward(const framework::ExecutionContext &ctx,
                                   const Tensor *fmha_or_gate_out, int m, int n,
                                   int k) {
  auto *out_linear_weight = ctx.Input<Tensor>("OutLinearWeight");
  auto *out_linear_bias = ctx.Input<Tensor>("OutLinearBias");

  auto *out = ctx.Output<Tensor>("Y");
  out->mutable_data<T>(ctx.GetPlace());

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
                                    int m, int n, int k, bool is_gating) {
  auto *d_out = ctx.Input<Tensor>(framework::GradVarName("Y"));
  auto *out_linear_weight = ctx.Input<Tensor>("OutLinearWeight");
  auto *input = is_gating ? ctx.Input<Tensor>("GateOut") : fmha_out;

  auto *d_out_linear_weight =
      ctx.Output<Tensor>(framework::GradVarName("OutLinearWeight"));
  auto *d_out_linear_bias =
      ctx.Output<Tensor>(framework::GradVarName("OutLinearBias"));
  auto *d_input = is_gating
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
    // x: qkv's input [batch_size, seq_len, dim_embed]
    // y: qkv's weight: [3, num_head, dim_head, dim_embed]
    auto *x = ctx.Input<Tensor>("X");
    auto *src_mask = ctx.Input<Tensor>("SrcMask");
    auto *nonbatched_bias = ctx.Input<Tensor>("NonbatchedBias");

    auto *qkv_transpose_out = ctx.Output<Tensor>("QKVTransposeOut");
    auto *qk_out = ctx.Output<Tensor>("QKOut");
    auto *softmax_out = ctx.Output<Tensor>("SoftmaxOut");
    auto *qktv_out = ctx.Output<Tensor>("QKTVOut");
    // auto *fmha_out = ctx.Output<Tensor>("FMHAOut");

    qkv_transpose_out->mutable_data<T>(ctx.GetPlace());
    qk_out->mutable_data<T>(ctx.GetPlace());
    qktv_out->mutable_data<T>(ctx.GetPlace());
    softmax_out->mutable_data<T>(ctx.GetPlace());
    // fmha_out->mutable_data<T>(ctx.GetPlace());

    const auto is_gating = ctx.Attr<bool>("is_gating");

    const auto x_dims = x->dims();
    const auto qkv_w_dims = ctx.Input<Tensor>("QKVWeight")->dims();

    int batch_size = x_dims[0];
    int seq_len_m = x_dims[1];
    int seq_len_r = x_dims[2];
    int hidden_size = x_dims[3];  // qkv_dim

    // qkv_weight[3, n_head, c, qkv_dim]
    int num_head = qkv_w_dims[1];
    int c = qkv_w_dims[2];

    // 1. Merged QKV Matmul: einsum(nbhqk,nbkhc -> nbqhc)
    //    [batch_size * seq_len_m * seq_len_r * 3 * num_head * c]
    int m = batch_size * seq_len_m * seq_len_r;
    int n = 3 * num_head * c;
    int k = hidden_size;
    Tensor *qkv_out = ComputeMergedQKVMatmulForward<T>(ctx, x, m, n, k);

    // 2. FMHA
    Tensor fmha_out;
    fmha_out.Resize({batch_size, seq_len_m, seq_len_r, num_head, c});
    fmha_out.mutable_data<T>(ctx.GetPlace());
    auto fmha_compute = FMHAGateRef<T>(ctx.cuda_device_context(), batch_size,
                                       seq_len_m, seq_len_r, num_head, c);
    fmha_compute.ComputeForward(nonbatched_bias, *qkv_out, src_mask,
                                qkv_transpose_out, qk_out, softmax_out,
                                qktv_out, &fmha_out);

    // 3. Gating Linear
    Tensor *fmha_or_gate_out = nullptr;
    if (is_gating) {
      m = batch_size * seq_len_m * seq_len_r;
      n = num_head * c;
      k = hidden_size;
      fmha_or_gate_out =
          ComputeGatingLinearForward<T>(ctx, x, &fmha_out, m, n, k);
    } else {
      fmha_or_gate_out = &fmha_out;
    }

    // 4. Output Linear
    m = batch_size * seq_len_m * seq_len_r;
    n = hidden_size;
    k = num_head * c;
    ComputeOutputLinearForward<T>(ctx, fmha_or_gate_out, m, n, k);
  }
};

template <typename T>
class FusedGateAttentionGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto is_gating = ctx.Attr<bool>("is_gating");

    // fw input
    auto *x = ctx.Input<Tensor>("X");
    auto *src_mask = ctx.Input<Tensor>("SrcMask");
    auto *nonbatched_bias = ctx.Input<Tensor>("NonbatchedBias");

    // fw output
    auto *qkv_transpose_out = ctx.Input<Tensor>("QKVTransposeOut");
    auto *qk_out = ctx.Input<Tensor>("QKOut");
    auto *softmax_out = ctx.Input<Tensor>("SoftmaxOut");
    auto *qktv_out = ctx.Input<Tensor>("QKTVOut");

    // output's grad
    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_qkv_out = ctx.Output<Tensor>(framework::GradVarName("QKVOut"));
    auto *d_qktv_out = ctx.Output<Tensor>(framework::GradVarName("QKTVOut"));
    auto *d_qkv_transpose_out =
        ctx.Output<Tensor>(framework::GradVarName("QKVTransposeOut"));
    auto *d_qk_out = ctx.Output<Tensor>(framework::GradVarName("QKOut"));
    auto *d_softmax_out =
        ctx.Output<Tensor>(framework::GradVarName("SoftmaxOut"));

    d_x->mutable_data<T>(ctx.GetPlace());
    d_qkv_out->mutable_data<T>(ctx.GetPlace());
    d_qktv_out->mutable_data<T>(ctx.GetPlace());
    d_qkv_transpose_out->mutable_data<T>(ctx.GetPlace());
    d_qk_out->mutable_data<T>(ctx.GetPlace());
    d_softmax_out->mutable_data<T>(ctx.GetPlace());

    // parameter grad
    auto *d_nonbatched_bias =
        ctx.Output<Tensor>(framework::GradVarName("NonbatchedBias"));
    if (nonbatched_bias) {
      d_nonbatched_bias->mutable_data<T>(ctx.GetPlace());
    }

    const auto x_dims = x->dims();
    const auto qkv_w_dims = ctx.Input<Tensor>("QKVWeight")->dims();

    int batch_size = x_dims[0];
    int seq_len_m = x_dims[1];
    int seq_len_r = x_dims[2];
    int hidden_size = x_dims[3];

    // qkv_weight[3, n_head, c, qkv_dim]
    int num_head = qkv_w_dims[1];
    int c = qkv_w_dims[2];

    // Re-compute the fmha_out.
    Tensor fmha_out;
    fmha_out.Resize({batch_size, seq_len_m, seq_len_r, num_head, c});
    fmha_out.mutable_data<T>(ctx.GetPlace());

    auto fmha_compute = FMHAGateRef<T>(ctx.cuda_device_context(), batch_size,
                                       seq_len_m, seq_len_r, num_head, c);
    fmha_compute.ComputeQKTVTransposeForward(*qktv_out, &fmha_out);

    // 1. Gradient of Output Linear
    int m = batch_size * seq_len_m * seq_len_r;
    int n = hidden_size;
    int k = num_head * c;

    Tensor d_fmha_out;
    d_fmha_out.Resize({batch_size, seq_len_m, seq_len_r, num_head, c});
    d_fmha_out.mutable_data<T>(ctx.GetPlace());

    Tensor *d_fhma_or_gate_out = ComputeOutputLinearBackward<T>(
        ctx, &fmha_out, &d_fmha_out, m, n, k, is_gating);

    // 2. Gradient of Gating Linear
    if (is_gating) {
      m = batch_size * seq_len_m * seq_len_r;
      n = num_head * c;
      k = hidden_size;
      // d_fhma_or_gate_out is d_gate_out.
      ComputeGatingLinearBackward<T>(ctx, &fmha_out, d_fhma_or_gate_out, d_x,
                                     &d_fmha_out, m, n, k);
    }

    // 3. Gradient of FMHA
    fmha_compute.ComputeBackward(
        *qkv_transpose_out, src_mask, *softmax_out, *qk_out, d_fmha_out,
        nonbatched_bias, d_nonbatched_bias, d_qktv_out, d_softmax_out, d_qk_out,
        nullptr, d_qkv_transpose_out, d_qkv_out);

    // 4. Gradient of Merged QKV Matmul
    m = batch_size * seq_len_m * seq_len_r;
    n = 3 * num_head * c;
    k = hidden_size;

    bool use_addto = true;
    if (use_addto) {
      ComputeMergedQKVMatmulBackward<T>(ctx, x, d_qkv_out, d_x, m, n, k, true);
    } else {
      Tensor d_residual;
      d_residual.Resize(x_dims);
      d_residual.mutable_data<T>(ctx.GetPlace());
      ComputeMergedQKVMatmulBackward<T>(ctx, x, d_qkv_out, &d_residual, m, n, k,
                                        false);

      // Gradient accumulation
      std::vector<const Tensor *> ins = {&d_residual, d_x};
      std::vector<Tensor *> outs = {d_x};
      paddle::operators::LaunchSameDimsElementwiseCudaKernel<T>(
          ctx.cuda_device_context(), ins, &outs, AddFunctor<T>());
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(fused_gate_attention,
                        ops::FusedGateAttentionOpKernel<float>,
                        ops::FusedGateAttentionOpKernel<double>,
                        ops::FusedGateAttentionOpKernel<plat::float16>);
REGISTER_OP_CUDA_KERNEL(fused_gate_attention_grad,
                        ops::FusedGateAttentionGradKernel<float>,
                        ops::FusedGateAttentionGradKernel<double>,
                        ops::FusedGateAttentionGradKernel<plat::float16>);
