/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the Licnse. */

#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/batch_norm_op.h"
#include "paddle/fluid/platform/collective_helper.h"
#if defined(PADDLE_WITH_CNCL)
#include "paddle/fluid/platform/device/mlu/cncl_helper.h"
#endif
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

#define NO_USE_CNCL 0
#define GET_LAYOUT_OFFSET 2

using Tensor = phi::DenseTensor;
static std::vector<cnnlTensorLayout_t> supported_input_layout = {
    CNNL_LAYOUT_NC, CNNL_LAYOUT_NLC, CNNL_LAYOUT_NHWC, CNNL_LAYOUT_NDHWC};

template <typename T>
class SyncBatchNormMLUKernel : public framework::OpKernel<T> {
  using MPDType = typename details::MPTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    float epsilon = ctx.Attr<float>("epsilon");
    float momentum = ctx.Attr<float>("momentum");
    const bool is_test = ctx.Attr<bool>("is_test");
    const bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    const bool trainable_stats = ctx.Attr<bool>("trainable_statistics");
    const std::string layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout layout = framework::StringToDataLayout(layout_str);

    PADDLE_ENFORCE_EQ(use_global_stats,
                      false,
                      platform::errors::InvalidArgument(
                          "sync_batch_norm doesn't support "
                          "to set use_global_stats True. Please use batch_norm "
                          "in this case."));

    const auto *x = ctx.Input<phi::DenseTensor>("X");
    const auto *scale = ctx.Input<phi::DenseTensor>("Scale");
    const auto *bias = ctx.Input<phi::DenseTensor>("Bias");
    const auto *mean = ctx.Input<phi::DenseTensor>("Mean");
    const auto *variance = ctx.Input<phi::DenseTensor>("Variance");
    auto *mean_out = ctx.Output<phi::DenseTensor>("MeanOut");
    auto *variance_out = ctx.Output<phi::DenseTensor>("VarianceOut");
    auto *saved_mean = ctx.Output<phi::DenseTensor>("SavedMean");
    auto *saved_variance = ctx.Output<phi::DenseTensor>("SavedVariance");
    auto *y = ctx.Output<phi::DenseTensor>("Y");

    const auto &x_dims = x->dims();
    PADDLE_ENFORCE_GE(x_dims.size(),
                      2,
                      platform::errors::InvalidArgument(
                          "The Input dim size should be larger than 1."));
    PADDLE_ENFORCE_LE(x_dims.size(),
                      5,
                      platform::errors::InvalidArgument(
                          "The Input dim size should be less than 6."));

    int N, C, H, W, D;
    ExtractNCWHD(x_dims, layout, &N, &C, &H, &W, &D);

    y->mutable_data<T>(ctx.GetPlace());
    mean_out->mutable_data<MPDType>(ctx.GetPlace());
    variance_out->mutable_data<MPDType>(ctx.GetPlace());
    saved_mean->mutable_data<MPDType>(ctx.GetPlace());
    saved_variance->mutable_data<MPDType>(ctx.GetPlace());

    Tensor trans_x;
    Tensor trans_y;
    std::vector<int> forward_perm;
    std::vector<int> backward_perm;
    std::vector<int> trans_shape;
    const bool need_transpose =
        ((layout == DataLayout::kNCHW && x_dims.size() != 2) ||
         x_dims.size() == 5);
    if (need_transpose) {
      SetMLUTransposePerm(
          x_dims, layout, &forward_perm, &backward_perm, &trans_shape);
      trans_x.mutable_data<T>(phi::make_ddim(trans_shape), ctx.GetPlace());
      trans_y.mutable_data<T>(phi::make_ddim(trans_shape), ctx.GetPlace());
      MLUCnnlTensorDesc desc_x(*x);
      MLUCnnlTensorDesc desc_trans_x(
          trans_shape.size(), trans_shape.data(), ToCnnlDataType(x->dtype()));
      MLUCnnl::Transpose(ctx,
                         forward_perm,
                         x_dims.size(),
                         desc_x.get(),
                         GetBasePtr(x),
                         desc_trans_x.get(),
                         GetBasePtr(&trans_x));
    } else {
      trans_x = *x;
      trans_y = *y;
    }

    MLUCnnlTensorDesc desc_trans(
        trans_x,
        supported_input_layout[x_dims.size() - GET_LAYOUT_OFFSET],
        ToCnnlDataType<T>());

    bool test_mode = is_test && (!trainable_stats);
    if (test_mode) {  // inference
      MLUCnnlTensorDesc desc_weight_bias_mean_var(*bias);
      MLUCnnl::FusedBatchNorm(ctx,
                              false /*is_training*/,
                              desc_trans.get(),
                              GetBasePtr(&trans_x),
                              desc_weight_bias_mean_var.get(),
                              GetBasePtr(scale),
                              GetBasePtr(bias),
                              GetBasePtr(mean),
                              GetBasePtr(variance),
                              epsilon,
                              momentum,
                              desc_trans.get(),
                              GetBasePtr(&trans_y),
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr);
    } else {  // training
      if (ctx.HasInput("MomentumTensor")) {
        const auto *mom_tensor = ctx.Input<phi::DenseTensor>("MomentumTensor");
        Tensor mom_cpu;
        paddle::framework::TensorCopySync(
            *mom_tensor, platform::CPUPlace(), &mom_cpu);
        momentum = mom_cpu.data<float>()[0];
      }

      Tensor local_mean, local_var;
      local_mean.mutable_data<MPDType>(mean->dims(), ctx.GetPlace());
      local_var.mutable_data<MPDType>(variance->dims(), ctx.GetPlace());
      MLUCnnlTensorDesc desc_mean_var(*mean_out);

      // cacl local_mean and local_var
      MLUCnnl::SyncBatchNormStats(ctx,
                                  desc_trans.get(),
                                  GetBasePtr(&trans_x),
                                  epsilon,
                                  desc_mean_var.get(),
                                  GetBasePtr(&local_mean),
                                  desc_mean_var.get(),
                                  GetBasePtr(&local_var));

      Tensor input_count;
      input_count.mutable_data<MPDType>(phi::make_ddim({1}), ctx.GetPlace());
      FillMLUTensorWithHostValue<MPDType>(
          ctx, static_cast<MPDType>(x->numel() / C), &input_count);

      Tensor count_all;
      Tensor mean_all(mean->dtype());
      Tensor invstd_all(variance->dtype());

#ifdef PADDLE_WITH_CNCL
      auto &dev_ctx =
          ctx.template device_context<paddle::platform::MLUDeviceContext>();
      auto *comm = dev_ctx.cncl_comm();
      if (comm) {
        auto cncl_comm = paddle::platform::CNCLCommContext::Instance().Get(
            0, ctx.GetPlace());
        auto *comm = cncl_comm->comm();
        auto comm_stream = cncl_comm->stream();
        int count;
        PADDLE_ENFORCE_MLU_SUCCESS(cnclGetCommCount(&count, comm));
        count_all.mutable_data<MPDType>(phi::make_ddim({count}),
                                        ctx.GetPlace());
        mean_all.mutable_data<MPDType>(phi::make_ddim({count, mean->numel()}),
                                       ctx.GetPlace());
        invstd_all.mutable_data<MPDType>(
            phi::make_ddim({count, variance->numel()}), ctx.GetPlace());
        // before comm_stream exec, need sync compute_stream.
        dev_ctx.Wait();

        cnclDataType_t dtype = platform::ToCNCLDataType(
            framework::TransToProtoVarType(count_all.dtype()));
        PADDLE_ENFORCE_MLU_SUCCESS(cnclAllGather(GetBasePtr(&input_count),
                                                 GetBasePtr(&count_all),
                                                 1,
                                                 dtype,
                                                 comm,
                                                 comm_stream));

        auto cncl_dtype = platform::ToCNCLDataType(
            framework::TransToProtoVarType(mean_all.dtype()));
        PADDLE_ENFORCE_MLU_SUCCESS(cnclAllGather(GetBasePtr(&local_mean),
                                                 GetBasePtr(&mean_all),
                                                 local_mean.numel(),
                                                 cncl_dtype,
                                                 comm,
                                                 comm_stream));

        PADDLE_ENFORCE_MLU_SUCCESS(cnclAllGather(GetBasePtr(&local_var),
                                                 GetBasePtr(&invstd_all),
                                                 local_var.numel(),
                                                 cncl_dtype,
                                                 comm,
                                                 comm_stream));
        // after comm_stream exec, need sync queue for using compute_stream
        // correctly.
        PADDLE_ENFORCE_MLU_SUCCESS(cnrtQueueSync(comm_stream));
#else
      if (NO_USE_CNCL) {
#endif
      } else {
        count_all = input_count;
        mean_all.ShareDataWith(local_mean);
        invstd_all.ShareDataWith(local_var);
        mean_all.Resize(phi::make_ddim({1, local_mean.numel()}));
        invstd_all.Resize(phi::make_ddim({1, local_var.numel()}));
      }

      MLUCnnlTensorDesc desc_all_mean_invstd(
          invstd_all, CNNL_LAYOUT_NC, ToCnnlDataType<MPDType>());
      MLUCnnlTensorDesc desc_moving_mean_var(*mean_out);
      MLUCnnlTensorDesc desc_saved_mean_var(*saved_mean);
      MLUCnnlTensorDesc desc_count_all(count_all);

      MLUCnnl::SyncBatchNormGatherStatsWithCounts(ctx,
                                                  momentum,
                                                  epsilon,
                                                  desc_all_mean_invstd.get(),
                                                  GetBasePtr(&mean_all),
                                                  desc_all_mean_invstd.get(),
                                                  GetBasePtr(&invstd_all),
                                                  desc_moving_mean_var.get(),
                                                  GetBasePtr(mean_out),
                                                  desc_moving_mean_var.get(),
                                                  GetBasePtr(variance_out),
                                                  desc_count_all.get(),
                                                  GetBasePtr(&count_all),
                                                  desc_saved_mean_var.get(),
                                                  GetBasePtr(saved_mean),
                                                  desc_saved_mean_var.get(),
                                                  GetBasePtr(saved_variance));

      MLUCnnlTensorDesc desc_other_param(*saved_mean);
      MLUCnnl::SyncBatchNormElemt(ctx,
                                  desc_trans.get(),
                                  GetBasePtr(&trans_x),
                                  desc_other_param.get(),
                                  GetBasePtr(saved_mean),
                                  desc_other_param.get(),
                                  GetBasePtr(saved_variance),
                                  desc_other_param.get(),
                                  GetBasePtr(scale),
                                  desc_other_param.get(),
                                  GetBasePtr(bias),
                                  desc_trans.get(),
                                  GetBasePtr(&trans_y));
    }
    if (need_transpose) {
      MLUCnnlTensorDesc desc_y(*y);
      MLUCnnlTensorDesc desc_trans_y(trans_y);
      MLUCnnl::Transpose(ctx,
                         backward_perm,
                         trans_y.dims().size(),
                         desc_trans_y.get(),
                         GetBasePtr(&trans_y),
                         desc_y.get(),
                         GetBasePtr(y));
    }
  }
};

template <typename T>
class SyncBatchNormMLUGradKernel : public framework::OpKernel<T> {
  using MPDType = typename details::MPTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const std::string layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout layout = framework::StringToDataLayout(layout_str);

    const auto *d_y = ctx.Input<phi::DenseTensor>(framework::GradVarName("Y"));
    const auto *scale = ctx.Input<phi::DenseTensor>("Scale");
    const auto *bias = ctx.Input<phi::DenseTensor>("Bias");

    // init output
    auto *d_x = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto *d_scale =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<phi::DenseTensor>(framework::GradVarName("Bias"));

    const auto *saved_mean = ctx.Input<phi::DenseTensor>("SavedMean");
    const auto *saved_inv_var = ctx.Input<phi::DenseTensor>("SavedVariance");

    const Tensor *x;
    if (ctx.HasInput("Y")) {
      PADDLE_ENFORCE_EQ(true,
                        false,
                        platform::errors::InvalidArgument(
                            "sync_batch_norm_grad doesn't support input Y"));
    } else {
      x = ctx.Input<phi::DenseTensor>("X");
    }

    const auto &x_dims = x->dims();
    PADDLE_ENFORCE_GE(x_dims.size(),
                      2,
                      platform::errors::InvalidArgument(
                          "The Input X dim size should be larger than 1."));
    PADDLE_ENFORCE_LE(x_dims.size(),
                      5,
                      platform::errors::InvalidArgument(
                          "The Input X dim size should be less than 6."));

    int N, C, H, W, D;
    ExtractNCWHD(x_dims, layout, &N, &C, &H, &W, &D);
    PADDLE_ENFORCE_EQ(scale->dims()[0],
                      C,
                      platform::errors::InvalidArgument(
                          "Expected first dim for input parameter(scale) of "
                          "OP(sync_batch_norm) be (%d), but given (%d).",
                          C,
                          scale->dims()[0]));

    d_x->mutable_data<T>(ctx.GetPlace());
    if (d_scale && d_bias) {
      d_scale->mutable_data<MPDType>(ctx.GetPlace());
      d_bias->mutable_data<MPDType>(ctx.GetPlace());
    }
    PADDLE_ENFORCE_EQ(scale->dims().size(),
                      1UL,
                      platform::errors::InvalidArgument(
                          "Expected rank for input parameter(scale) of "
                          "OP(sync_batch_norm) be (1), but given (%d).",
                          scale->dims().size()));

    Tensor trans_x;
    Tensor trans_dy;
    Tensor trans_dx;
    std::vector<int> forward_perm;
    std::vector<int> backward_perm;
    std::vector<int> trans_shape;
    const bool need_transpose =
        ((layout == DataLayout::kNCHW && x_dims.size() != 2) ||
         x_dims.size() == 5);
    if (need_transpose) {
      SetMLUTransposePerm(
          x_dims, layout, &forward_perm, &backward_perm, &trans_shape);
      trans_x.mutable_data<T>(phi::make_ddim(trans_shape), ctx.GetPlace());
      trans_dy.mutable_data<T>(phi::make_ddim(trans_shape), ctx.GetPlace());
      trans_dx.mutable_data<T>(phi::make_ddim(trans_shape), ctx.GetPlace());
      MLUCnnlTensorDesc desc_x(*x);
      MLUCnnlTensorDesc desc_trans_x(
          trans_shape.size(), trans_shape.data(), ToCnnlDataType(x->dtype()));
      MLUCnnl::Transpose(ctx,
                         forward_perm,
                         x_dims.size(),
                         desc_x.get(),
                         GetBasePtr(x),
                         desc_trans_x.get(),
                         GetBasePtr(&trans_x));
      MLUCnnl::Transpose(ctx,
                         forward_perm,
                         x_dims.size(),
                         desc_x.get(),
                         GetBasePtr(d_y),
                         desc_trans_x.get(),
                         GetBasePtr(&trans_dy));
    } else {
      trans_x = *x;
      trans_dy = *d_y;
      trans_dx = *d_x;
    }
    MLUCnnlTensorDesc desc_trans(
        trans_x,
        supported_input_layout[x_dims.size() - GET_LAYOUT_OFFSET],
        ToCnnlDataType<T>());

    Tensor sum_dy, sum_dy_xmu;
    sum_dy.mutable_data<MPDType>(bias->dims(), ctx.GetPlace());
    sum_dy_xmu.mutable_data<MPDType>(bias->dims(), ctx.GetPlace());
    MLUCnnlTensorDesc desc_other_param(*bias);

    MLUCnnl::SyncBatchnormBackwardReduce(
        ctx,
        desc_trans.get(),
        GetBasePtr(&trans_dy),
        desc_trans.get(),
        GetBasePtr(&trans_x),
        desc_other_param.get(),
        GetBasePtr(saved_mean),
        desc_other_param.get(),
        GetBasePtr(saved_inv_var),
        d_scale ? desc_other_param.get() : nullptr,
        d_scale ? GetBasePtr(d_scale) : nullptr,
        d_bias ? desc_other_param.get() : nullptr,
        d_bias ? GetBasePtr(d_bias) : nullptr,
        desc_other_param.get(),
        GetBasePtr(&sum_dy),
        desc_other_param.get(),
        GetBasePtr(&sum_dy_xmu),
        true /*compute sum_dy, sum_dy_xmu*/,
        d_scale ? true : false /*compute d_scale*/,
        d_bias ? true : false /*compute d_bias*/);

    Tensor numel_count;
    numel_count.mutable_data<int32_t>(phi::make_ddim({1}), ctx.GetPlace());
    FillMLUTensorWithHostValue<int32_t>(
        ctx, static_cast<int32_t>(x->numel() / C), &numel_count);

#ifdef PADDLE_WITH_CNCL
    auto &dev_ctx =
        ctx.template device_context<paddle::platform::MLUDeviceContext>();
    auto *comm = dev_ctx.cncl_comm();
    if (comm) {
      auto cncl_comm =
          paddle::platform::CNCLCommContext::Instance().Get(0, ctx.GetPlace());
      auto *comm = cncl_comm->comm();
      auto comm_stream = cncl_comm->stream();
      // before comm_stream exec, need sync compute_stream.
      dev_ctx.Wait();
      cnclDataType_t dtype = platform::ToCNCLDataType(
          framework::TransToProtoVarType(numel_count.dtype()));
      PADDLE_ENFORCE_MLU_SUCCESS(cnclAllReduce(GetBasePtr(&numel_count),
                                               GetBasePtr(&numel_count),
                                               1,
                                               dtype,
                                               cnclSum,
                                               comm,
                                               comm_stream));

      auto cncl_dtype = platform::ToCNCLDataType(
          framework::TransToProtoVarType(sum_dy.dtype()));
      PADDLE_ENFORCE_MLU_SUCCESS(cnclAllReduce(GetBasePtr(&sum_dy),
                                               GetBasePtr(&sum_dy),
                                               sum_dy.numel(),
                                               cncl_dtype,
                                               cnclSum,
                                               comm,
                                               comm_stream));

      PADDLE_ENFORCE_MLU_SUCCESS(cnclAllReduce(GetBasePtr(&sum_dy_xmu),
                                               GetBasePtr(&sum_dy_xmu),
                                               sum_dy_xmu.numel(),
                                               cncl_dtype,
                                               cnclSum,
                                               comm,
                                               comm_stream));
      // after comm_stream exec, need sync queue for using compute_stream
      // correctly.
      PADDLE_ENFORCE_MLU_SUCCESS(cnrtQueueSync(comm_stream));
    }
#endif

    if (d_x) {
      MLUCnnlTensorDesc desc_count(numel_count);
      MLUCnnl::SyncBatchNormBackwardElemt(ctx,
                                          desc_trans.get(),
                                          GetBasePtr(&trans_dy),
                                          desc_trans.get(),
                                          GetBasePtr(&trans_x),
                                          desc_other_param.get(),
                                          GetBasePtr(saved_mean),
                                          desc_other_param.get(),
                                          GetBasePtr(saved_inv_var),
                                          desc_other_param.get(),
                                          GetBasePtr(scale),
                                          desc_other_param.get(),
                                          GetBasePtr(&sum_dy),
                                          desc_other_param.get(),
                                          GetBasePtr(&sum_dy_xmu),
                                          desc_count.get(),
                                          GetBasePtr(&numel_count),
                                          desc_trans.get(),
                                          GetBasePtr(&trans_dx));

      if (need_transpose) {
        MLUCnnlTensorDesc desc_dx(*d_x);
        MLUCnnlTensorDesc desc_trans_dx(trans_dx);
        MLUCnnl::Transpose(ctx,
                           backward_perm,
                           trans_dx.dims().size(),
                           desc_trans_dx.get(),
                           GetBasePtr(&trans_dx),
                           desc_dx.get(),
                           GetBasePtr(d_x));
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_MLU_KERNEL(sync_batch_norm,
                       ops::SyncBatchNormMLUKernel<float>,
                       ops::SyncBatchNormMLUKernel<plat::float16>);

REGISTER_OP_MLU_KERNEL(sync_batch_norm_grad,
                       ops::SyncBatchNormMLUGradKernel<float>,
                       ops::SyncBatchNormMLUGradKernel<plat::float16>);
