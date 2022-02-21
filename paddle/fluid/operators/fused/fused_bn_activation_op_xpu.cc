/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_XPU

#include "paddle/fluid/operators/fused/fused_bn_activation_op.h"
#include <memory>
#include <string>
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/batch_norm_op.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DDim = framework::DDim;
using DataLayout = framework::DataLayout;

template <typename DeviceContext, typename T>
class FusedBatchNormActXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_xpu_place(ctx.GetPlace()), true,
        platform::errors::PreconditionNotMet("The Place should be XPU"));
    using XPU_T = typename XPUTypeTrait<T>::Type;
    const auto epsilon = ctx.Attr<float>("epsilon");
    const auto momentum = ctx.Attr<float>("momentum");
    const auto is_test = ctx.Attr<bool>("is_test");
    const auto use_global_stats = ctx.Attr<bool>("use_global_stats");
    const auto trainable_stats = ctx.Attr<bool>("trainable_statistics");
    bool test_mode = is_test && (!trainable_stats);
    bool global_stats = test_mode || use_global_stats;

    const auto* x = ctx.Input<Tensor>("X");
    const auto& x_dims = x->dims();
    PADDLE_ENFORCE_EQ(x_dims.size(), 4,
                      platform::errors::InvalidArgument(
                          "The input tensor X's dimension must equal to 4. But "
                          "received X's shape = [%s], X's dimension = [%d].",
                          x_dims, x_dims.size()));
    auto data_layout = framework::StringToDataLayout("NCHW");
    int N, C, H, W, D;
    ExtractNCWHD(x_dims, data_layout, &N, &C, &H, &W, &D);
    const auto* scale = ctx.Input<Tensor>("Scale");
    const auto* bias = ctx.Input<Tensor>("Bias");
    auto* x_data = reinterpret_cast<const XPU_T*>(x->data<T>());
    const auto* scale_data = scale->data<float>();
    const auto* bias_data = bias->data<float>();
    auto* y = ctx.Output<Tensor>("Y");
    y->mutable_data<T>(ctx.GetPlace());
    auto* y_data = reinterpret_cast<XPU_T*>(y->data<T>());
    const auto act_type = ctx.Attr<std::string>("act_type");
    bool act_type_check = (act_type == "relu") || (act_type == "linear");
    PADDLE_ENFORCE_EQ(act_type_check, 1,
                      platform::errors::InvalidArgument(
                          "The activation type has to be relu or linear."));
    auto* mean_out = ctx.Output<Tensor>("MeanOut");
    auto* variance_out = ctx.Output<Tensor>("VarianceOut");
    auto* saved_mean = ctx.Output<Tensor>("SavedMean");
    auto* saved_variance = ctx.Output<Tensor>("SavedVariance");
    mean_out->mutable_data<float>(ctx.GetPlace());
    variance_out->mutable_data<float>(ctx.GetPlace());
    saved_mean->mutable_data<float>(ctx.GetPlace());
    saved_variance->mutable_data<float>(ctx.GetPlace());
    auto* mean_out_data = mean_out->data<float>();
    auto* variance_out_data = variance_out->data<float>();
    auto* saved_mean_data = saved_mean->data<float>();
    auto* saved_variance_data = saved_variance->data<float>();

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());

    int r = XPU_SUCCESS;
    if (!global_stats) {
      auto xpu_act_type = act_type == "relu" ? xpu::Activation_t::RELU
                                             : xpu::Activation_t::LINEAR;
      r = xpu::batch_norm_fusion<XPU_T>(
          dev_ctx.x_context(), x_data, y_data, N, C, H, W, epsilon, momentum,
          scale_data, bias_data, saved_mean_data, saved_variance_data,
          mean_out_data, variance_out_data, true, nullptr, xpu_act_type,
          nullptr, 0);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "batch_norm_fusion");
    } else {
      const auto* mean = ctx.Input<Tensor>("Mean");
      const auto* variance = ctx.Input<Tensor>("Variance");
      const auto* mean_data = mean->data<float>();
      const auto* variance_data = variance->data<float>();
      if (act_type == "relu") {
        auto bn_out_data = RAII_GUARD.alloc<T>(x->numel());
        r = xpu::batch_norm_infer(dev_ctx.x_context(), x_data, bn_out_data, N,
                                  C, H, W, epsilon, scale_data, bias_data,
                                  mean_data, variance_data, true);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "batch_norm_infer");

        r = xpu::relu(dev_ctx.x_context(), bn_out_data, y_data, x->numel());
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "relu");
      } else {
        r = xpu::batch_norm_infer(dev_ctx.x_context(), x_data, y_data, N, C, H,
                                  W, epsilon, scale_data, bias_data, mean_data,
                                  variance_data, true);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "batch_norm_infer");
      }
    }
  }
};

template <typename DeviceContext, typename T>
class FusedBatchNormActGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using XPU_T = typename XPUTypeTrait<T>::Type;
    const auto* x = ctx.Input<Tensor>("X");
    const auto* y = ctx.Input<Tensor>("Y");
    const auto* dy = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto* scale = ctx.Input<Tensor>("Scale");
    const auto* saved_mean = ctx.Input<Tensor>("SavedMean");
    const auto* saved_inv_variance = ctx.Input<Tensor>("SavedVariance");

    const auto& x_dims = x->dims();
    PADDLE_ENFORCE_EQ(x_dims.size(), 4,
                      platform::errors::InvalidArgument(
                          "The input tensor X's dimension must equal to 4. But "
                          "received X's shape = [%s], X's dimension = [%d].",
                          x_dims, x_dims.size()));
    int N, C, H, W, D;
    auto data_layout = framework::StringToDataLayout("NCHW");
    ExtractNCWHD(x_dims, data_layout, &N, &C, &H, &W, &D);
    auto* x_data = reinterpret_cast<const XPU_T*>(x->data<T>());
    auto y_data = reinterpret_cast<const XPU_T*>(y->data<T>());
    auto* dy_data = reinterpret_cast<const XPU_T*>(dy->data<T>());
    const auto* scale_data = scale->data<float>();
    const auto* saved_mean_data = saved_mean->data<float>();
    const auto* saved_inv_variance_data = saved_inv_variance->data<float>();
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dscale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto* dbias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    XPU_T* dx_data = nullptr;
    float* dscale_data = nullptr;
    float* dbias_data = nullptr;

    if (dx != nullptr) {
      dx->mutable_data<T>(ctx.GetPlace());
      dx_data = reinterpret_cast<XPU_T*>(dx->data<T>());
    }
    if (dscale != nullptr) {
      dscale_data = dscale->mutable_data<float>(ctx.GetPlace());
    }
    if (dbias != nullptr) {
      dbias_data = dbias->mutable_data<float>(ctx.GetPlace());
    }
    const auto act_type = ctx.Attr<std::string>("act_type");
    bool act_type_check = (act_type == "relu") || (act_type == "linear");
    PADDLE_ENFORCE_EQ(act_type_check, 1,
                      platform::errors::InvalidArgument(
                          "The activation type has to be relu or linear."));

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto xpu_act_type = act_type == "relu" ? xpu::Activation_t::RELU
                                           : xpu::Activation_t::LINEAR;
    int r = xpu::batch_norm_grad_fusion<XPU_T>(
        dev_ctx.x_context(), x_data, y_data, dy_data, dx_data, N, C, H, W,
        scale_data, saved_mean_data, saved_inv_variance_data, dscale_data,
        dbias_data, true, nullptr, xpu_act_type, nullptr, 0);
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
        platform::errors::External("XPU API(fuse_batch_norm_act_grad) return "
                                   "wrong value[%d], please check whether "
                                   "Baidu Kunlun Card is properly installed.",
                                   r));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(
    fused_batch_norm_act,
    // ops::FusedBatchNormActXPUKernel<paddle::platform::XPUDeviceContext,
    // paddle::platform::float16>,
    ops::FusedBatchNormActXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    fused_batch_norm_act_grad,
    // ops::FusedBatchNormActGradXPUKernel<paddle::platform::XPUDeviceContext,
    // paddle::platform::float16>,
    ops::FusedBatchNormActGradXPUKernel<paddle::platform::XPUDeviceContext,
                                        float>);

#endif  // PADDLE_WITH_XPU
