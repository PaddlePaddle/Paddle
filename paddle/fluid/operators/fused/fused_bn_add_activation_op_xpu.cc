// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_XPU

#include "paddle/fluid/operators/fused/fused_bn_add_activation_op.h"
#include "paddle/fluid/operators/norm_utils.h"
#include "paddle/fluid/platform/device/device_wrapper.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DDim = framework::DDim;
using DataLayout = framework::DataLayout;

template <typename DeviceContext, typename T>
class FusedBatchNormAddActXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_xpu_place(ctx.GetPlace()), true,
        platform::errors::PreconditionNotMet("It must use XPUPlace."));
    float epsilon = static_cast<float>(ctx.Attr<float>("epsilon"));
    float momentum = ctx.Attr<float>("momentum");
    epsilon = std::max<float>(epsilon, 1e-5);

    const auto is_test = ctx.Attr<bool>("is_test");
    const auto use_global_stats = ctx.Attr<bool>("use_global_stats");
    const auto trainable_stats = ctx.Attr<bool>("trainable_statistics");
    bool test_mode = is_test && (!trainable_stats);
    bool global_stats = test_mode || use_global_stats;

    const auto *x = ctx.Input<Tensor>("X");
    const auto *z = ctx.Input<Tensor>("Z");
    const auto &in_dims = x->dims();

    auto data_layout = x->layout();
    PADDLE_ENFORCE_EQ(data_layout, DataLayout::kNCHW,
                      platform::errors::InvalidArgument(
                          "The 'data_layout' attribute must be NCHW. But "
                          "recevived 'data_layout' is [%s].",
                          framework::DataLayoutToString(data_layout)));

    std::string act_type = ctx.Attr<std::string>("act_type");
    bool act_type_check = (act_type == "relu") || (act_type == "linear");
    PADDLE_ENFORCE_EQ(act_type_check, 1,
                      platform::errors::InvalidArgument(
                          "The activation type has to be relu or linear."));

    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");

    auto *mean_out = ctx.Output<Tensor>("MeanOut");
    auto *variance_out = ctx.Output<Tensor>("VarianceOut");
    mean_out->mutable_data<float>(ctx.GetPlace());
    variance_out->mutable_data<float>(ctx.GetPlace());

    auto *saved_mean = ctx.Output<Tensor>("SavedMean");
    auto *saved_variance = ctx.Output<Tensor>("SavedVariance");
    saved_mean->mutable_data<float>(ctx.GetPlace());
    saved_variance->mutable_data<float>(ctx.GetPlace());

    auto *y = ctx.Output<Tensor>("Y");
    y->mutable_data<T>(ctx.GetPlace());

    int N, C, H, W, D;
    ExtractNCWHD(in_dims, DataLayout::kNCHW, &N, &C, &H, &W, &D);

    auto &dev_ctx = ctx.template device_context<platform::XPUDeviceContext>();
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());

    const auto *x_data = x->data<T>();
    const auto *z_data = z->data<T>();
    const auto *scale_data = scale->data<float>();
    const auto *bias_data = bias->data<float>();
    auto *y_data = y->data<T>();

    int r = XPU_SUCCESS;
    auto xpu_act_type = act_type == "relu" ? xpu::Activation_t::RELU
                                           : xpu::Activation_t::LINEAR;
    if (!global_stats) {
      auto *mean_out_data = mean_out->data<float>();
      auto *variance_out_data = variance_out->data<float>();
      auto *saved_mean_data = saved_mean->data<float>();
      auto *saved_variance_data = saved_variance->data<float>();
      r = xpu::batch_norm_fusion<T>(
          dev_ctx.x_context(), x_data, y_data, N, C, H, W, epsilon, momentum,
          scale_data, bias_data, saved_mean_data, saved_variance_data,
          mean_out_data, variance_out_data, true, z_data, xpu_act_type, nullptr,
          0);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "batch_norm_fusion");
    } else {
      // batch_norm_infer + add + relu
      const auto *mean = ctx.Input<Tensor>("Mean");
      const auto *variance = ctx.Input<Tensor>("Variance");
      const auto *mean_data = mean->data<float>();
      const auto *variance_data = variance->data<float>();

      auto tmp_out_data = RAII_GUARD.alloc<T>(x->numel());
      r = xpu::batch_norm_infer(dev_ctx.x_context(), x_data, y_data, N, C, H, W,
                                epsilon, scale_data, bias_data, mean_data,
                                variance_data, true);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "batch_norm_infer");

      r = xpu::add(dev_ctx.x_context(), z_data, y_data, tmp_out_data,
                   x->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");

      r = xpu::relu(dev_ctx.x_context(), tmp_out_data, y_data, x->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "relu");
    }
  }
};

template <typename DeviceContext, typename T>
class FusedBatchNormAddActGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_xpu_place(ctx.GetPlace()), true,
        platform::errors::PreconditionNotMet("It must use XPUPlace."));
    std::string act_type = ctx.Attr<std::string>("act_type");

    const auto *x = ctx.Input<Tensor>("X");
    const auto *y = ctx.Input<Tensor>("Y");
    const auto *dy = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *saved_mean = ctx.Input<Tensor>("SavedMean");
    const auto *saved_var = ctx.Input<Tensor>("SavedVariance");
    const auto &in_dims = x->dims();

    int N, C, H, W, D;
    ExtractNCWHD(in_dims, DataLayout::kNCHW, &N, &C, &H, &W, &D);

    auto *dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *dz = ctx.Output<Tensor>(framework::GradVarName("Z"));
    auto *dscale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *dbias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    dx->mutable_data<T>(ctx.GetPlace());
    dz->mutable_data<T>(ctx.GetPlace());
    PADDLE_ENFORCE_EQ(
        dscale && dbias, true,
        platform::errors::PreconditionNotMet(
            "Both the scale grad and the bias grad must not be null."));
    dscale->mutable_data<float>(ctx.GetPlace());
    dbias->mutable_data<float>(ctx.GetPlace());
    PADDLE_ENFORCE_EQ(scale->dims().size(), 1UL,
                      platform::errors::PreconditionNotMet(
                          "The scale only has one dimension."));
    PADDLE_ENFORCE_EQ(
        scale->dims()[0], C,
        platform::errors::PreconditionNotMet(
            "The size of scale is equal to the channel of Input(X)."));

    auto &dev_ctx = ctx.template device_context<platform::XPUDeviceContext>();
    const auto *x_data = x->data<T>();
    const auto *y_data = y->data<T>();
    const auto *dy_data = dy->data<T>();
    const auto *scale_data = scale->data<T>();
    const auto *saved_mean_data = saved_mean->data<float>();
    const auto *saved_var_data = saved_var->data<float>();
    auto *dx_data = dx->data<T>();
    auto *dz_data = dz->data<T>();
    auto *dscale_data = dscale->data<float>();
    auto *dbias_data = dbias->data<float>();

    auto xpu_act_type = act_type == "relu" ? xpu::Activation_t::RELU
                                           : xpu::Activation_t::LINEAR;
    int r = xpu::batch_norm_grad_fusion<T>(
        dev_ctx.x_context(), x_data, y_data, dy_data, dx_data, N, C, H, W,
        scale_data, saved_mean_data, saved_var_data, dscale_data, dbias_data,
        true, dz_data, xpu_act_type, nullptr, 0);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "batch_norm_grad_fusion");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_XPU_KERNEL(
    fused_bn_add_activation,
    ops::FusedBatchNormAddActXPUKernel<plat::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    fused_bn_add_activation_grad,
    ops::FusedBatchNormAddActGradXPUKernel<plat::XPUDeviceContext, float>);

#endif
