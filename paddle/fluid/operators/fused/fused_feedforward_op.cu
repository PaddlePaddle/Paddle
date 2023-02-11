/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/fused/fused_dropout_helper.h"
#include "paddle/fluid/operators/layer_norm_kernel.cu.h"
#include "paddle/fluid/operators/matmul_v2_op.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/distributed/collective/process_group_nccl.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
static void AllReduce(phi::DenseTensor& tensor,  // NOLINT
                      const int ring_id,
                      const phi::GPUContext& ctx) {
  if (ring_id == -1) return;
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();

  if (map->has(ring_id)) {
    paddle::distributed::ProcessGroup* pg = map->get(ring_id);
    auto pg_nccl = static_cast<distributed::ProcessGroupNCCL*>(pg);
    paddle::distributed::AllreduceOptions opts;
    opts.reduce_op = distributed::ReduceOp::SUM;
    auto task = pg_nccl->AllReduce(&tensor, tensor, opts, true, true);
    task->Wait();
  } else {
    auto dtype = platform::ToNCCLDataType(
        framework::TransToProtoVarType(tensor.dtype()));
    int64_t numel = tensor.numel();
    const void* sendbuff = tensor.data<T>();
    auto place = ctx.GetPlace();
    void* recvbuff = ctx.Alloc<T>(&tensor, tensor.numel() * sizeof(T));
    auto comm = platform::NCCLCommContext::Instance().Get(ring_id, place);
    auto stream = ctx.stream();
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
        sendbuff, recvbuff, numel, dtype, ncclSum, comm->comm(), stream));
  }
#else
  PADDLE_THROW(platform::errors::Unimplemented(
      "PaddlePaddle should compile with NCCL or RCCL when used tensor model "
      "parallel op."));
#endif
}

template <typename DeviceContext, typename T>
class FusedFeedForwardKernel : public framework::OpKernel<T> {
 public:
  void MatMul(const phi::GPUContext& ctx,
              const phi::DenseTensor& a,
              const phi::DenseTensor& b,
              phi::DenseTensor* c) const {
    auto blas = phi::funcs::GetBlas<DeviceContext, T>(ctx);
    auto a_2d = FoldInitDims(a);
    auto b_2d = FoldInitDims(b);
    auto mat_dim_a = phi::funcs::CreateMatrixDescriptor(a_2d.dims(), 0, false);
    auto mat_dim_b = phi::funcs::CreateMatrixDescriptor(b_2d.dims(), 0, false);
    T alpha = static_cast<T>(1.0);
    blas.MatMul(a, mat_dim_a, b, mat_dim_b, alpha, c, T(0));
  }

  void FFN(const phi::GPUContext& ctx,
           const phi::DenseTensor& x,
           const phi::DenseTensor& linear1_weight,
           const phi::DenseTensor* linear1_bias,
           const phi::DenseTensor& linear2_weight,
           const phi::DenseTensor* linear2_bias,
           const phi::DenseTensor* ln1_scale,
           const phi::DenseTensor* ln1_bias,
           const phi::DenseTensor* ln2_scale,
           const phi::DenseTensor* ln2_bias,
           phi::DenseTensor* out,
           phi::DenseTensor* dropout1_mask,
           phi::DenseTensor* dropout2_mask,
           phi::DenseTensor* ln1_mean,
           phi::DenseTensor* ln1_variance,
           phi::DenseTensor* ln2_mean,
           phi::DenseTensor* ln2_variance,
           phi::DenseTensor* linear1_out,
           phi::DenseTensor* ln1_out,
           phi::DenseTensor* dropout1_out,
           phi::DenseTensor* dropout2_out,
           const int bsz_seq,
           const int d_model,
           const int dim_feedforward,
           const std::string& act_method,
           const bool pre_layer_norm,
           const float epsilon1,
           const float epsilon2,
           const bool add_residual,
           const int ring_id,
           const DropoutParam& dropout_param1,
           const DropoutParam& dropout_param2) const {
    FusedDropoutLayerNormHelper<T, uint8_t> pre_layernorm_helper(
        bsz_seq, d_model, epsilon1);
    FusedDropoutHelper<T, uint8_t> fused_act_dropout_helper(
        ctx, bsz_seq, dim_feedforward, dropout_param1);
    FusedDropoutLayerNormHelper<T, uint8_t> fused_dropout_layernorm_helper(
        ctx, bsz_seq, d_model, dropout_param2, epsilon2);

    using U = LayerNormParamType<T>;
    const phi::DenseTensor* in = &x;

    const U* ln1_scale_ptr =
        ln1_scale == nullptr ? nullptr : ln1_scale->data<U>();
    const U* ln1_bias_ptr = ln1_bias == nullptr ? nullptr : ln1_bias->data<U>();
    const U* ln2_scale_ptr =
        ln2_scale == nullptr ? nullptr : ln2_scale->data<U>();
    const U* ln2_bias_ptr = ln2_bias == nullptr ? nullptr : ln2_bias->data<U>();
    const T* linear1_bias_ptr =
        linear1_bias == nullptr ? nullptr : linear1_bias->data<T>();
    const T* linear2_bias_ptr =
        linear2_bias == nullptr ? nullptr : linear2_bias->data<T>();

    if (pre_layer_norm) {
      pre_layernorm_helper.LayerNorm(ctx,
                                     x.data<T>(),
                                     ln1_scale_ptr,
                                     ln1_bias_ptr,
                                     ln1_out->data<T>(),
                                     ln1_mean->data<U>(),
                                     ln1_variance->data<U>());
      in = ln1_out;
    }
    MatMul(ctx, *in, linear1_weight, linear1_out);
    fused_act_dropout_helper.DropoutActBias(ctx,
                                            linear1_out->data<T>(),
                                            linear1_bias_ptr,
                                            act_method,
                                            dropout1_out->data<T>(),
                                            dropout1_mask->data<uint8_t>());
    phi::DenseTensor linear2_out;
    linear2_out.Resize({bsz_seq, d_model});
    ctx.Alloc<T>(&linear2_out, linear2_out.numel() * sizeof(T));
    MatMul(ctx, *dropout1_out, linear2_weight, &linear2_out);

    // tensor model parallel
    AllReduce<T>(linear2_out, ring_id, ctx);

    const T* residual_ptr = add_residual ? x.data<T>() : nullptr;
    if (!pre_layer_norm) {
      // TODO(Xreki): support post layer_norm case when add_residual is false.
      PADDLE_ENFORCE_EQ(add_residual,
                        true,
                        platform::errors::InvalidArgument(
                            "Attribute add_residual is expected to be true "
                            "when pre_layer_norm is false."));

      fused_dropout_layernorm_helper.LayernormResidualDropoutBias(
          ctx,
          linear2_out.data<T>(),
          residual_ptr,
          linear2_bias_ptr,
          ln2_scale_ptr,
          ln2_bias_ptr,
          dropout2_out->data<T>(),
          dropout2_mask->data<uint8_t>(),
          out->data<T>(),
          ln2_mean->data<U>(),
          ln2_variance->data<U>());
    } else {
      fused_dropout_layernorm_helper.ResidualDropoutBias(
          ctx,
          linear2_out.data<T>(),
          residual_ptr,
          linear2_bias_ptr,
          out->data<T>(),
          dropout2_mask->data<uint8_t>());
    }
  }

  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<phi::DenseTensor>("X");
    auto* linear1_weight = context.Input<phi::DenseTensor>("Linear1Weight");
    auto* linear1_bias = context.Input<phi::DenseTensor>("Linear1Bias");
    auto* linear2_weight = context.Input<phi::DenseTensor>("Linear2Weight");
    auto* linear2_bias = context.Input<phi::DenseTensor>("Linear2Bias");
    const bool pre_layer_norm = context.Attr<bool>("pre_layer_norm");
    auto& dev_ctx = context.template device_context<phi::GPUContext>();

    auto* ln1_scale =
        pre_layer_norm ? context.Input<phi::DenseTensor>("Ln1Scale") : nullptr;
    auto* ln1_bias =
        pre_layer_norm ? context.Input<phi::DenseTensor>("Ln1Bias") : nullptr;
    auto* ln2_scale =
        !pre_layer_norm ? context.Input<phi::DenseTensor>("Ln2Scale") : nullptr;
    auto* ln2_bias =
        !pre_layer_norm ? context.Input<phi::DenseTensor>("Ln2Bias") : nullptr;

    auto* ln1_mean =
        pre_layer_norm ? context.Output<phi::DenseTensor>("Ln1Mean") : nullptr;
    auto* ln1_variance = pre_layer_norm
                             ? context.Output<phi::DenseTensor>("Ln1Variance")
                             : nullptr;
    auto* ln2_mean =
        !pre_layer_norm ? context.Output<phi::DenseTensor>("Ln2Mean") : nullptr;
    auto* ln2_variance = !pre_layer_norm
                             ? context.Output<phi::DenseTensor>("Ln2Variance")
                             : nullptr;
    auto* out = context.Output<phi::DenseTensor>("Out");
    auto* dropout1_mask = context.Output<phi::DenseTensor>("Dropout1Mask");
    auto* dropout2_mask = context.Output<phi::DenseTensor>("Dropout2Mask");
    auto* linear1_out = context.Output<phi::DenseTensor>("Linear1Out");
    auto* ln1_out =
        pre_layer_norm ? context.Output<phi::DenseTensor>("Ln1Out") : nullptr;
    auto* dropout1_out = context.Output<phi::DenseTensor>("Dropout1Out");
    auto* dropout2_out = context.Output<phi::DenseTensor>("Dropout2Out");

    const std::string act_method = context.Attr<std::string>("act_method");

    const float epsilon1 = context.Attr<float>("ln1_epsilon");
    const float epsilon2 = context.Attr<float>("ln2_epsilon");
    const int ring_id = context.Attr<int>("ring_id");
    const bool add_residual = context.Attr<bool>("add_residual");

    DropoutParam dropout_param1(context, 1);
    DropoutParam dropout_param2(context, 2);

    using U = LayerNormParamType<T>;
    dev_ctx.Alloc<T>(out, out->numel() * sizeof(T));
    dev_ctx.Alloc<uint8_t>(dropout1_mask,
                           dropout1_mask->numel() * sizeof(uint8_t));
    dev_ctx.Alloc<uint8_t>(dropout2_mask,
                           dropout2_mask->numel() * sizeof(uint8_t));
    if (pre_layer_norm) {
      dev_ctx.Alloc<U>(ln1_mean, ln1_mean->numel() * sizeof(U));
      dev_ctx.Alloc<U>(ln1_variance, ln1_variance->numel() * sizeof(U));
      dev_ctx.Alloc<T>(ln1_out, ln1_out->numel() * sizeof(T));
    } else {
      dev_ctx.Alloc<U>(ln2_mean, ln2_mean->numel() * sizeof(U));
      dev_ctx.Alloc<U>(ln2_variance, ln2_variance->numel() * sizeof(U));
    }

    dev_ctx.Alloc<T>(linear1_out, linear1_out->numel() * sizeof(T));
    dev_ctx.Alloc<T>(dropout1_out, dropout1_out->numel() * sizeof(T));
    dev_ctx.Alloc<T>(dropout2_out, dropout2_out->numel() * sizeof(T));

    auto x_dim = x->dims();
    auto mat_dim_x = phi::funcs::CreateMatrixDescriptor(
        RowMatrixFromVector(x_dim), 0, false);

    auto dim = linear1_weight->dims();
    int d_model = dim[0];
    int dim_feedforward = dim[dim.size() - 1];
    int bsz_seq = mat_dim_x.batch_size_ * mat_dim_x.height_;

    FFN(context.cuda_device_context(),
        *x,
        *linear1_weight,
        linear1_bias,
        *linear2_weight,
        linear2_bias,
        ln1_scale,
        ln1_bias,
        ln2_scale,
        ln2_bias,
        out,
        dropout1_mask,
        dropout2_mask,
        ln1_mean,
        ln1_variance,
        ln2_mean,
        ln2_variance,
        linear1_out,
        ln1_out,
        dropout1_out,
        dropout2_out,
        bsz_seq,
        d_model,
        dim_feedforward,
        act_method,
        pre_layer_norm,
        epsilon1,
        epsilon2,
        add_residual,
        ring_id,
        dropout_param1,
        dropout_param2);
  }
};

template <typename DeviceContext, typename T>
class FusedFeedForwardGradKernel : public framework::OpKernel<T> {
 public:
  void MatMulGrad(const phi::GPUContext& ctx,
                  const phi::DenseTensor& d_out,
                  const phi::DenseTensor& a,
                  const phi::DenseTensor& b,
                  phi::DenseTensor* d_a,
                  phi::DenseTensor* d_b) const {
    auto blas = phi::funcs::GetBlas<DeviceContext, T>(ctx);
    auto a_2d = FoldInitDims(a);
    auto b_2d = FoldInitDims(b);
    auto mat_dim_a = phi::funcs::CreateMatrixDescriptor(a_2d.dims(), 0, true);
    auto mat_dim_b = phi::funcs::CreateMatrixDescriptor(b_2d.dims(), 0, true);
    auto mat_dim_dout =
        phi::funcs::CreateMatrixDescriptor(d_out.dims(), 0, false);
    T alpha = static_cast<T>(1.0);
    blas.MatMul(d_out, mat_dim_dout, b, mat_dim_b, alpha, d_a, T(0));
    blas.MatMul(a, mat_dim_a, d_out, mat_dim_dout, alpha, d_b, T(0));
  }

  void FFNGrad(const phi::GPUContext& ctx,
               const phi::DenseTensor& d_out,
               const phi::DenseTensor& x,
               const phi::DenseTensor& dropout1_mask,
               const phi::DenseTensor& dropout2_mask,
               const phi::DenseTensor& linear1_out,
               const phi::DenseTensor* ln1_out,
               const phi::DenseTensor& dropout1_out,
               const phi::DenseTensor* dropout2_out,
               const phi::DenseTensor& linear1_weight,
               const phi::DenseTensor* linear1_bias,
               const phi::DenseTensor& linear2_weight,
               const phi::DenseTensor* ln1_gamma,
               const phi::DenseTensor* ln1_beta,
               const phi::DenseTensor* ln1_mean,
               const phi::DenseTensor* ln1_variance,
               const phi::DenseTensor* ln2_gamma,
               const phi::DenseTensor* ln2_beta,
               const phi::DenseTensor* ln2_mean,
               const phi::DenseTensor* ln2_variance,
               phi::DenseTensor* d_x,
               phi::DenseTensor* d_linear1_weight,
               phi::DenseTensor* d_linear1_bias,
               phi::DenseTensor* d_linear2_weight,
               phi::DenseTensor* d_linear2_bias,
               phi::DenseTensor* d_ln1_gamma,
               phi::DenseTensor* d_ln1_beta,
               phi::DenseTensor* d_ln2_gamma,
               phi::DenseTensor* d_ln2_beta,
               const int bsz_seq,
               const int d_model,
               const int dim_feedforward,
               const DropoutParam& dropout_param1,
               const DropoutParam& dropout_param2,
               const std::string& act_method,
               const bool pre_layer_norm,
               const float epsilon1,
               const float epsilon2,
               const bool add_residual,
               const int ring_id) const {
    FusedDropoutLayerNormHelper<T, uint8_t> pre_layernorm_helper(
        bsz_seq, d_model, epsilon1);
    FusedDropoutHelper<T, uint8_t> fused_act_dropout_helper(
        ctx, bsz_seq, dim_feedforward, dropout_param1);
    FusedDropoutLayerNormHelper<T, uint8_t> fused_dropout_layernorm_helper(
        ctx, bsz_seq, d_model, dropout_param2, epsilon2);

    using U = LayerNormParamType<T>;
    const U* ln1_gamma_ptr =
        ln1_gamma == nullptr ? nullptr : ln1_gamma->data<U>();
    const U* ln1_beta_ptr = ln1_beta == nullptr ? nullptr : ln1_beta->data<U>();
    const U* ln2_gamma_ptr =
        ln2_gamma == nullptr ? nullptr : ln2_gamma->data<U>();
    const U* ln2_beta_ptr = ln2_beta == nullptr ? nullptr : ln2_beta->data<U>();
    const T* linear1_bias_ptr =
        linear1_bias == nullptr ? nullptr : linear1_bias->data<T>();
    T* d_linear1_bias_ptr =
        d_linear1_bias == nullptr ? nullptr : d_linear1_bias->data<T>();
    T* d_linear2_bias_ptr =
        d_linear2_bias == nullptr ? nullptr : d_linear2_bias->data<T>();
    U* d_ln1_gamma_ptr =
        d_ln1_gamma == nullptr ? nullptr : d_ln1_gamma->data<U>();
    U* d_ln1_beta_ptr = d_ln1_beta == nullptr ? nullptr : d_ln1_beta->data<U>();
    U* d_ln2_gamma_ptr =
        d_ln2_gamma == nullptr ? nullptr : d_ln2_gamma->data<U>();
    U* d_ln2_beta_ptr = d_ln2_beta == nullptr ? nullptr : d_ln2_beta->data<U>();

    phi::DenseTensor d_linear2_out, d_dropout2_out, d_residual;
    d_linear2_out.Resize({bsz_seq, d_model});
    ctx.Alloc<T>(&d_linear2_out, d_linear2_out.numel() * sizeof(T));
    d_dropout2_out.Resize({bsz_seq, d_model});
    ctx.Alloc<T>(&d_dropout2_out, d_dropout2_out.numel() * sizeof(T));

    T* d_residual_ptr = nullptr;
    if (add_residual) {
      d_residual.Resize(d_x->dims());
      d_residual_ptr =
          ctx.Alloc<T>(&d_residual, d_residual.numel() * sizeof(T));
    }
    if (pre_layer_norm) {
      fused_dropout_layernorm_helper.ResidualDropoutBiasGrad(
          ctx,
          d_out.data<T>(),
          dropout2_mask.data<uint8_t>(),
          d_linear2_out.data<T>(),
          d_residual_ptr,
          d_linear2_bias_ptr);
    } else {
      fused_dropout_layernorm_helper.LayernormResidualDropoutBiasGrad(
          ctx,
          d_out.data<T>(),
          dropout2_out->data<T>(),
          dropout2_mask.data<uint8_t>(),
          ln2_gamma_ptr,
          ln2_mean->data<U>(),
          ln2_variance->data<U>(),
          d_dropout2_out.data<T>(),
          d_ln2_gamma_ptr,
          d_ln2_beta_ptr,
          d_linear2_out.data<T>(),
          d_linear2_bias_ptr,
          d_residual_ptr);
    }

    phi::DenseTensor d_dropout1_out;
    d_dropout1_out.Resize({bsz_seq, dim_feedforward});
    ctx.Alloc<T>(&d_dropout1_out, d_dropout1_out.numel() * sizeof(T));
    MatMulGrad(ctx,
               d_linear2_out,
               dropout1_out,
               linear2_weight,
               &d_dropout1_out,
               d_linear2_weight);

    phi::DenseTensor d_linear1_out;
    d_linear1_out.Resize({bsz_seq, dim_feedforward});
    ctx.Alloc<T>(&d_linear1_out, d_linear1_out.numel() * sizeof(T));
    fused_act_dropout_helper.DropoutActBiasGrad(ctx,
                                                d_dropout1_out.data<T>(),
                                                linear1_out.data<T>(),
                                                linear1_bias_ptr,
                                                dropout1_mask.data<uint8_t>(),
                                                d_linear1_out.data<T>(),
                                                d_linear1_bias_ptr,
                                                act_method);

    if (pre_layer_norm) {
      phi::DenseTensor d_ln1_out;
      d_ln1_out.Resize({bsz_seq, d_model});
      ctx.Alloc<T>(&d_ln1_out, d_ln1_out.numel() * sizeof(T));
      MatMulGrad(ctx,
                 d_linear1_out,
                 *ln1_out,
                 linear1_weight,
                 &d_ln1_out,
                 d_linear1_weight);
      // tensor model parallel
      AllReduce<T>(d_ln1_out, ring_id, ctx);
      pre_layernorm_helper.LayerNormGrad(ctx,
                                         d_ln1_out.data<T>(),
                                         x.data<T>(),
                                         ln1_gamma_ptr,
                                         ln1_mean->data<U>(),
                                         ln1_variance->data<U>(),
                                         d_x->data<T>(),
                                         d_ln1_gamma_ptr,
                                         d_ln1_beta_ptr);
    } else {
      MatMulGrad(ctx, d_linear1_out, x, linear1_weight, d_x, d_linear1_weight);
      // tensor model parallel
      AllReduce<T>(*d_x, ring_id, ctx);
    }

    if (add_residual) {
      // gradient accumulation
      std::vector<const phi::DenseTensor*> ins = {&d_residual, d_x};
      std::vector<phi::DenseTensor*> outs = {d_x};
      phi::funcs::ElementwiseKernel<T>(
          ctx, ins, &outs, phi::funcs::AddFunctor<T>());
    }
  }

  void Compute(const framework::ExecutionContext& context) const override {
    using U = LayerNormParamType<T>;
    auto& dev_ctx = context.template device_context<phi::GPUContext>();
    auto d_out =
        *context.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto x = *context.Input<phi::DenseTensor>("X");
    const bool pre_layer_norm = context.Attr<bool>("pre_layer_norm");
    auto dropout1_mask = *context.Input<phi::DenseTensor>("Dropout1Mask");
    auto dropout2_mask = *context.Input<phi::DenseTensor>("Dropout2Mask");
    auto linear1_out = *context.Input<phi::DenseTensor>("Linear1Out");
    auto* ln1_out =
        pre_layer_norm ? context.Input<phi::DenseTensor>("Ln1Out") : nullptr;
    auto dropout1_out = *context.Input<phi::DenseTensor>("Dropout1Out");
    auto* dropout2_out = context.Input<phi::DenseTensor>("Dropout2Out");
    auto linear1_weight = *context.Input<phi::DenseTensor>("Linear1Weight");
    auto* linear1_bias = context.Input<phi::DenseTensor>("Linear1Bias");
    auto linear2_weight = *context.Input<phi::DenseTensor>("Linear2Weight");
    auto* ln1_mean =
        pre_layer_norm ? context.Input<phi::DenseTensor>("Ln1Mean") : nullptr;
    auto* ln1_variance = pre_layer_norm
                             ? context.Input<phi::DenseTensor>("Ln1Variance")
                             : nullptr;
    auto* ln1_scale =
        pre_layer_norm ? context.Input<phi::DenseTensor>("Ln1Scale") : nullptr;
    auto* ln1_bias =
        pre_layer_norm ? context.Input<phi::DenseTensor>("Ln1Bias") : nullptr;
    auto* ln2_mean =
        !pre_layer_norm ? context.Input<phi::DenseTensor>("Ln2Mean") : nullptr;
    auto* ln2_variance = !pre_layer_norm
                             ? context.Input<phi::DenseTensor>("Ln2Variance")
                             : nullptr;
    auto* ln2_scale =
        !pre_layer_norm ? context.Input<phi::DenseTensor>("Ln2Scale") : nullptr;
    auto* ln2_bias =
        !pre_layer_norm ? context.Input<phi::DenseTensor>("Ln2Bias") : nullptr;

    auto* d_x = context.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* d_ln1_scale = pre_layer_norm ? context.Output<phi::DenseTensor>(
                                             framework::GradVarName("Ln1Scale"))
                                       : nullptr;
    auto* d_ln1_bias = pre_layer_norm ? context.Output<phi::DenseTensor>(
                                            framework::GradVarName("Ln1Bias"))
                                      : nullptr;
    auto* d_ln2_scale = pre_layer_norm
                            ? nullptr
                            : context.Output<phi::DenseTensor>(
                                  framework::GradVarName("Ln2Scale"));
    auto* d_ln2_bias = pre_layer_norm ? nullptr
                                      : context.Output<phi::DenseTensor>(
                                            framework::GradVarName("Ln2Bias"));
    auto* d_linear1_weight = context.Output<phi::DenseTensor>(
        framework::GradVarName("Linear1Weight"));
    auto* d_linear1_bias =
        context.Output<phi::DenseTensor>(framework::GradVarName("Linear1Bias"));
    auto* d_linear2_weight = context.Output<phi::DenseTensor>(
        framework::GradVarName("Linear2Weight"));
    auto* d_linear2_bias =
        context.Output<phi::DenseTensor>(framework::GradVarName("Linear2Bias"));

    const float epsilon1 = context.Attr<float>("ln1_epsilon");
    const float epsilon2 = context.Attr<float>("ln2_epsilon");
    const bool add_residual = context.Attr<bool>("add_residual");
    const int ring_id = context.Attr<int>("ring_id");
    const std::string act_method = context.Attr<std::string>("act_method");
    DropoutParam dropout_param1(context, 1);
    DropoutParam dropout_param2(context, 2);

    dev_ctx.Alloc<T>(d_x, d_x->numel() * sizeof(T));
    if (d_ln1_scale) {
      dev_ctx.Alloc<U>(d_ln1_scale, d_ln1_scale->numel() * sizeof(U));
    }
    if (d_ln1_bias) {
      dev_ctx.Alloc<U>(d_ln1_bias, d_ln1_bias->numel() * sizeof(U));
    }
    if (d_ln2_scale) {
      dev_ctx.Alloc<U>(d_ln2_scale, d_ln2_scale->numel() * sizeof(U));
    }
    if (d_ln2_bias) {
      dev_ctx.Alloc<U>(d_ln2_bias, d_ln2_bias->numel() * sizeof(U));
    }
    if (d_linear1_bias) {
      dev_ctx.Alloc<T>(d_linear1_bias, d_linear1_bias->numel() * sizeof(T));
    }
    if (d_linear2_bias) {
      dev_ctx.Alloc<T>(d_linear2_bias, d_linear2_bias->numel() * sizeof(T));
    }
    dev_ctx.Alloc<T>(d_linear1_weight, d_linear1_weight->numel() * sizeof(T));
    dev_ctx.Alloc<T>(d_linear2_weight, d_linear2_weight->numel() * sizeof(T));

    auto x_dim = x.dims();
    auto mat_dim_x = phi::funcs::CreateMatrixDescriptor(
        RowMatrixFromVector(x_dim), 0, false);

    auto linear1_weight_dim = linear1_weight.dims();
    int d_model = linear1_weight_dim[0];
    int dim_feedforward = linear1_weight_dim[linear1_weight_dim.size() - 1];
    int bsz_seq = mat_dim_x.batch_size_ * mat_dim_x.height_;

    FFNGrad(context.cuda_device_context(),
            d_out,
            x,
            dropout1_mask,
            dropout2_mask,
            linear1_out,
            ln1_out,
            dropout1_out,
            dropout2_out,
            linear1_weight,
            linear1_bias,
            linear2_weight,
            ln1_scale,
            ln1_bias,
            ln1_mean,
            ln1_variance,
            ln2_scale,
            ln2_bias,
            ln2_mean,
            ln2_variance,
            d_x,
            d_linear1_weight,
            d_linear1_bias,
            d_linear2_weight,
            d_linear2_bias,
            d_ln1_scale,
            d_ln1_bias,
            d_ln2_scale,
            d_ln2_bias,
            bsz_seq,
            d_model,
            dim_feedforward,
            dropout_param1,
            dropout_param2,
            act_method,
            pre_layer_norm,
            epsilon1,
            epsilon2,
            add_residual,
            ring_id);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    fused_feedforward,
    ops::FusedFeedForwardKernel<phi::GPUContext, float>,
    ops::FusedFeedForwardKernel<phi::GPUContext, double>,
    ops::FusedFeedForwardKernel<phi::GPUContext, paddle::platform::float16>);
REGISTER_OP_CUDA_KERNEL(
    fused_feedforward_grad,
    ops::FusedFeedForwardGradKernel<phi::GPUContext, float>,
    ops::FusedFeedForwardGradKernel<phi::GPUContext, double>,
    ops::FusedFeedForwardGradKernel<phi::GPUContext,
                                    paddle::platform::float16>);
