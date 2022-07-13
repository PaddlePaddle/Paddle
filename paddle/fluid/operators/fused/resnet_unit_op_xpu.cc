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
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

inline void fill_nchw(const std::vector<int> &dims,
                      bool is_nchw,
                      int *n,
                      int *c,
                      int *h,
                      int *w) {
  *n = dims[0];
  *c = dims[1];
  *h = dims[2];
  *w = dims[3];
  if (!is_nchw) {
    *c = dims[3];
    *h = dims[1];
    *w = dims[2];
  }
}

template <typename T>
class ResNetUnitXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_xpu_place(ctx.GetPlace()),
        true,
        platform::errors::PreconditionNotMet("It must use XPUPlace."));

    bool is_nchw = (ctx.Attr<std::string>("data_format") == "NCHW");
    // input x
    const Tensor *input_x = ctx.Input<Tensor>("X");
    const Tensor *filter_x = ctx.Input<Tensor>("FilterX");
    const Tensor *scale_x = ctx.Input<Tensor>("ScaleX");
    const Tensor *bias_x = ctx.Input<Tensor>("BiasX");

    // output x
    Tensor *conv_out_x = ctx.Output<Tensor>("ConvX");
    Tensor *saved_mean_x = ctx.Output<Tensor>("SavedMeanX");
    Tensor *saved_invstd_x = ctx.Output<Tensor>("SavedInvstdX");
    Tensor *running_mean_x = ctx.Output<Tensor>("RunningMeanX");
    Tensor *running_var_x = ctx.Output<Tensor>("RunningVarX");

    Tensor *output = ctx.Output<Tensor>("Y");

    //  attrs
    int padding = ctx.Attr<int>("padding");
    int stride = ctx.Attr<int>("stride");
    int stride_z = ctx.Attr<int>("stride_z");
    int dilation = ctx.Attr<int>("dilation");
    int group = ctx.Attr<int>("group");
    float eps = ctx.Attr<float>("epsilon");
    float momentum = ctx.Attr<float>("momentum");
    bool has_shortcut = ctx.Attr<bool>("has_shortcut");
    bool fuse_add = ctx.Attr<bool>("fuse_add");
    bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    bool is_test = ctx.Attr<bool>("is_test");
    bool is_train = !is_test && !use_global_stats;
    std::string act_type = ctx.Attr<std::string>("act_type");

    auto input_x_shape = phi::vectorize<int>(input_x->dims());
    auto filter_x_shape = phi::vectorize<int>(filter_x->dims());
    auto output_shape = phi::vectorize<int>(output->dims());

    auto place = ctx.GetPlace();
    auto output_ptr = output->mutable_data<T>(place);
    auto &dev_ctx = ctx.template device_context<platform::XPUDeviceContext>();

    std::vector<int> ksize = {filter_x_shape[2], filter_x_shape[3]};
    if (!is_nchw) {
      ksize[0] = filter_x_shape[1];
      ksize[1] = filter_x_shape[2];
    }
    std::vector<int> strides = {stride, stride};
    std::vector<int> paddings = {padding, padding};
    std::vector<int> dilations = {dilation, dilation};
    int conv_x_n = -1, conv_x_c = -1, conv_x_h = -1, conv_x_w = -1;
    fill_nchw(
        input_x_shape, is_nchw, &conv_x_n, &conv_x_c, &conv_x_h, &conv_x_w);

    // 1. Conv
    auto conv_out_x_ptr = conv_out_x->mutable_data<T>(place);
    int r = xpu::conv2d<T, T, T, int16_t>(dev_ctx.x_context(),
                                          input_x->data<T>(),
                                          filter_x->data<T>(),
                                          conv_out_x_ptr,
                                          conv_x_n,
                                          conv_x_c,
                                          conv_x_h,
                                          conv_x_w,
                                          filter_x_shape[0],
                                          ksize,
                                          strides,
                                          paddings,
                                          dilations,
                                          group,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          is_nchw);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "resnet unit conv");

    // 2. BN
    auto conv_out_x_shape = phi::vectorize<int>(conv_out_x->dims());
    Tensor bn_out;
    bn_out.Resize(conv_out_x->dims());
    auto bn_out_ptr = bn_out.mutable_data<T>(place);
    auto saved_mean_x_ptr = saved_mean_x->mutable_data<T>(place);
    auto saved_invstd_x_ptr = saved_invstd_x->mutable_data<T>(place);
    auto running_mean_x_ptr = running_mean_x->mutable_data<T>(place);
    auto running_var_x_ptr = running_var_x->mutable_data<T>(place);
    int bn_n = -1, bn_c = -1, bn_h = -1, bn_w = -1;
    fill_nchw(conv_out_x_shape, is_nchw, &bn_n, &bn_c, &bn_h, &bn_w);
    if (is_train) {
      r = xpu::batch_norm<T>(dev_ctx.x_context(),
                             conv_out_x_ptr,
                             bn_out_ptr,
                             bn_n,
                             bn_c,
                             bn_h,
                             bn_w,
                             eps,
                             momentum,
                             scale_x->data<T>(),
                             bias_x->data<T>(),
                             saved_mean_x_ptr,
                             saved_invstd_x_ptr,
                             running_mean_x_ptr,
                             running_var_x_ptr,
                             is_nchw);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "resnet unit bn");
    } else {
      r = xpu::batch_norm_infer<T>(dev_ctx.x_context(),
                                   conv_out_x_ptr,
                                   bn_out_ptr,
                                   bn_n,
                                   bn_c,
                                   bn_h,
                                   bn_w,
                                   eps,
                                   scale_x->data<T>(),
                                   bias_x->data<T>(),
                                   running_mean_x_ptr,
                                   running_var_x_ptr,
                                   is_nchw);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "resnet unit bn");
    }

    if (has_shortcut) {
      // input z
      const Tensor *input_z = ctx.Input<Tensor>("Z");
      const Tensor *filter_z = ctx.Input<Tensor>("FilterZ");
      const Tensor *scale_z = ctx.Input<Tensor>("ScaleZ");
      const Tensor *bias_z = ctx.Input<Tensor>("BiasZ");

      Tensor *conv_out_z = ctx.Output<Tensor>("ConvZ");
      Tensor *saved_mean_z = ctx.Output<Tensor>("SavedMeanZ");
      Tensor *saved_invstd_z = ctx.Output<Tensor>("SavedInvstdZ");
      Tensor *running_mean_z = ctx.Output<Tensor>("RunningMeanZ");
      Tensor *running_var_z = ctx.Output<Tensor>("RunningVarZ");

      auto input_z_shape = phi::vectorize<int>(input_z->dims());
      auto filter_z_shape = phi::vectorize<int>(filter_z->dims());

      // 3.1 Conv for second input
      std::vector<int> ksize_z = {filter_z_shape[2], filter_z_shape[3]};
      if (!is_nchw) {
        ksize_z[0] = filter_z_shape[1];
        ksize_z[1] = filter_z_shape[2];
      }
      std::vector<int> strides_z = {stride_z, stride_z};
      int conv_z_n = -1, conv_z_c = -1, conv_z_h = -1, conv_z_w = -1;
      fill_nchw(
          input_z_shape, is_nchw, &conv_z_n, &conv_z_c, &conv_z_h, &conv_z_w);

      // 1. Conv
      auto conv_out_z_ptr = conv_out_z->mutable_data<T>(place);
      r = xpu::conv2d<T, T, T, int16_t>(dev_ctx.x_context(),
                                        input_z->data<T>(),
                                        filter_z->data<T>(),
                                        conv_out_z_ptr,
                                        conv_z_n,
                                        conv_z_c,
                                        conv_z_h,
                                        conv_z_w,
                                        filter_z_shape[0],
                                        ksize_z,
                                        strides_z,
                                        paddings,
                                        dilations,
                                        group,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        is_nchw);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "resnet unit conv");

      // 3.2 BN for second input
      auto conv_out_z_shape = phi::vectorize<int>(conv_out_z->dims());
      Tensor bn_z_out;
      bn_z_out.Resize(conv_out_z->dims());
      auto bn_z_out_ptr = bn_z_out.mutable_data<T>(ctx.GetPlace());
      auto saved_mean_z_ptr = saved_mean_z->mutable_data<T>(ctx.GetPlace());
      auto saved_invstd_z_ptr = saved_invstd_z->mutable_data<T>(ctx.GetPlace());
      auto running_mean_z_ptr = running_mean_z->mutable_data<T>(ctx.GetPlace());
      auto running_var_z_ptr = running_var_z->mutable_data<T>(ctx.GetPlace());
      int bn_z_n = -1, bn_z_c = -1, bn_z_h = -1, bn_z_w = -1;
      fill_nchw(conv_out_z_shape, is_nchw, &bn_z_n, &bn_z_c, &bn_z_h, &bn_z_w);
      if (is_train) {
        r = xpu::batch_norm<T>(dev_ctx.x_context(),
                               conv_out_z_ptr,
                               bn_z_out_ptr,
                               bn_z_n,
                               bn_z_c,
                               bn_z_h,
                               bn_z_w,
                               eps,
                               momentum,
                               scale_z->data<T>(),
                               bias_z->data<T>(),
                               saved_mean_z_ptr,
                               saved_invstd_z_ptr,
                               running_mean_z_ptr,
                               running_var_z_ptr,
                               is_nchw);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "resnet unit bn");
      } else {
        r = xpu::batch_norm_infer<T>(dev_ctx.x_context(),
                                     conv_out_z_ptr,
                                     bn_z_out_ptr,
                                     bn_z_n,
                                     bn_z_c,
                                     bn_z_h,
                                     bn_z_w,
                                     eps,
                                     scale_z->data<T>(),
                                     bias_z->data<T>(),
                                     running_mean_z_ptr,
                                     running_var_z_ptr,
                                     is_nchw);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "resnet unit bn");
      }
      // 3.3 add + relu
      r = xpu::broadcast_add(dev_ctx.x_context(),
                             bn_out_ptr,
                             bn_z_out_ptr,
                             output_ptr,
                             conv_out_x_shape,
                             conv_out_z_shape);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "resnet unit add");
      r = xpu::relu(
          dev_ctx.x_context(), output_ptr, output_ptr, output->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "resnet unit relu");
    } else {
      if (fuse_add) {
        const Tensor *input_z = ctx.Input<Tensor>("Z");
        auto input_z_shape = phi::vectorize<int>(input_z->dims());
        r = xpu::broadcast_add(dev_ctx.x_context(),
                               bn_out_ptr,
                               input_z->data<T>(),
                               output_ptr,
                               conv_out_x_shape,
                               input_z_shape);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "resnet unit add");
        r = xpu::relu(
            dev_ctx.x_context(), output_ptr, output_ptr, output->numel());
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "resnet unit relu");
      } else {
        r = xpu::relu(
            dev_ctx.x_context(), bn_out_ptr, output_ptr, output->numel());
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "resnet unit relu");
      }
    }
  }
};

template <typename T>
class ResNetUnitGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto place = ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        platform::is_xpu_place(place),
        true,
        platform::errors::PreconditionNotMet("It must use XPUPlace."));

    bool is_nchw = (ctx.Attr<std::string>("data_format") == "NCHW");
    const Tensor *y_grad = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const Tensor *x = ctx.Input<Tensor>("X");
    const Tensor *filter_x = ctx.Input<Tensor>("FilterX");
    const Tensor *scale_x = ctx.Input<Tensor>("ScaleX");
    const Tensor *saved_mean_x = ctx.Input<Tensor>("SavedMeanX");
    const Tensor *saved_invstd_x = ctx.Input<Tensor>("SavedInvstdX");
    const Tensor *conv_out_x = ctx.Input<Tensor>("ConvX");
    const Tensor *output = ctx.Input<Tensor>("Y");

    Tensor *x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    Tensor *filter_x_grad =
        ctx.Output<Tensor>(framework::GradVarName("FilterX"));
    Tensor *scale_x_grad = ctx.Output<Tensor>(framework::GradVarName("ScaleX"));
    Tensor *bias_x_grad = ctx.Output<Tensor>(framework::GradVarName("BiasX"));

    auto x_grad_ptr = x_grad->mutable_data<T>(place);
    auto filter_x_grad_ptr = filter_x_grad->mutable_data<T>(place);
    auto scale_x_grad_ptr = scale_x_grad->mutable_data<T>(place);
    auto bias_x_grad_ptr = bias_x_grad->mutable_data<T>(place);

    int padding = ctx.Attr<int>("padding");
    int stride = ctx.Attr<int>("stride");
    int stride_z = ctx.Attr<int>("stride_z");
    int dilation = ctx.Attr<int>("dilation");
    int group = ctx.Attr<int>("group");
    float eps = ctx.Attr<float>("epsilon");
    bool has_shortcut = ctx.Attr<bool>("has_shortcut");
    bool fuse_add = ctx.Attr<bool>("fuse_add");
    std::string act_type = ctx.Attr<std::string>("act_type");

    std::vector<int> x_strides = {stride, stride};
    std::vector<int> z_strides = {stride, stride};
    std::vector<int> paddings = {padding, padding};
    std::vector<int> dilations = {dilation, dilation};

    auto x_shape = phi::vectorize<int>(x->dims());
    auto filter_x_shape = phi::vectorize<int>(filter_x->dims());
    auto output_shape = phi::vectorize<int>(output->dims());

    auto &dev_ctx = ctx.template device_context<platform::XPUDeviceContext>();

    Tensor conv_out_x_grad;
    conv_out_x_grad.Resize(conv_out_x->dims());
    conv_out_x_grad.mutable_data<T>(place);
    int r = 0;
    Tensor relu_grad;
    relu_grad.Resize(y_grad->dims());
    relu_grad.mutable_data<T>(place);
    auto relu_grad_ptr = relu_grad.data<T>();
    r = xpu::relu_grad(dev_ctx.x_context(),
                       output->data<T>(),
                       output->data<T>(),
                       y_grad->data<T>(),
                       relu_grad.data<T>(),
                       relu_grad.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "resnet_unit relu_grad");

    if (has_shortcut) {
      //       X                   Z
      //       |                   |
      //    NormConv            NormConv
      //       |                   |
      // BNStatsFinalize    BNStatsFinalize
      //       \                   /
      //          ScaleBiasAddRelu
      //                  |
      //                  Y
      const Tensor *z = ctx.Input<Tensor>("Z");
      const Tensor *filter_z = ctx.Input<Tensor>("FilterZ");
      const Tensor *scale_z = ctx.Input<Tensor>("ScaleZ");
      const Tensor *saved_mean_z = ctx.Input<Tensor>("SavedMeanZ");
      const Tensor *saved_invstd_z = ctx.Input<Tensor>("SavedInvstdZ");
      const Tensor *conv_out_z = ctx.Input<Tensor>("ConvZ");

      Tensor *z_grad = ctx.Output<Tensor>(framework::GradVarName("Z"));
      Tensor *filter_z_grad =
          ctx.Output<Tensor>(framework::GradVarName("FilterZ"));
      Tensor *scale_z_grad =
          ctx.Output<Tensor>(framework::GradVarName("ScaleZ"));
      Tensor *bias_z_grad = ctx.Output<Tensor>(framework::GradVarName("BiasZ"));

      auto z_grad_ptr = z_grad->mutable_data<T>(place);
      auto filter_z_grad_ptr = filter_z_grad->mutable_data<T>(place);
      auto scale_z_grad_ptr = scale_z_grad->mutable_data<T>(place);
      auto bias_z_grad_ptr = bias_z_grad->mutable_data<T>(place);

      auto conv_out_z_shape = phi::vectorize<int>(conv_out_z->dims());
      int bn_z_n = -1, bn_z_c = -1, bn_z_h = -1, bn_z_w = -1;
      fill_nchw(conv_out_z_shape, is_nchw, &bn_z_n, &bn_z_c, &bn_z_h, &bn_z_w);

      Tensor conv_out_z_grad;
      conv_out_z_grad.Resize(conv_out_z->dims());
      conv_out_z_grad.mutable_data<T>(place);
      r = xpu::batch_norm_grad<T>(dev_ctx.x_context(),
                                  conv_out_z->data<T>(),
                                  relu_grad_ptr,
                                  conv_out_z_grad.data<T>(),
                                  bn_z_n,
                                  bn_z_c,
                                  bn_z_h,
                                  bn_z_w,
                                  scale_z->data<T>(),
                                  saved_mean_z->data<T>(),
                                  saved_invstd_z->data<T>(),
                                  scale_z_grad_ptr,
                                  bias_z_grad_ptr,
                                  is_nchw,
                                  nullptr,
                                  nullptr,
                                  eps);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "resnet unit bn_grad");

      auto conv_out_x_shape = phi::vectorize<int>(conv_out_x->dims());
      int bn_x_n = -1, bn_x_c = -1, bn_x_h = -1, bn_x_w = -1;
      fill_nchw(conv_out_x_shape, is_nchw, &bn_x_n, &bn_x_c, &bn_x_h, &bn_x_w);
      r = xpu::batch_norm_grad<T>(dev_ctx.x_context(),
                                  conv_out_x->data<T>(),
                                  relu_grad_ptr,
                                  conv_out_x_grad.data<T>(),
                                  bn_x_n,
                                  bn_x_c,
                                  bn_x_h,
                                  bn_x_w,
                                  scale_x->data<T>(),
                                  saved_mean_x->data<T>(),
                                  saved_invstd_x->data<T>(),
                                  scale_x_grad_ptr,
                                  bias_x_grad_ptr,
                                  is_nchw,
                                  nullptr,
                                  nullptr,
                                  eps);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "resnet unit bn_grad");

      auto z_shape = phi::vectorize<int>(z->dims());
      auto filter_z_shape = phi::vectorize<int>(filter_z->dims());
      int z_n = -1, z_c = -1, z_h = -1, z_w = -1;
      fill_nchw(z_shape, is_nchw, &z_n, &z_c, &z_h, &z_w);
      std::vector<int> z_ksize = {filter_z_shape[2], filter_z_shape[3]};
      if (!is_nchw) {
        z_ksize[0] = filter_z_shape[1];
        z_ksize[1] = filter_z_shape[2];
      }
      std::vector<int> z_strides = {stride_z, stride_z};
      r = xpu::conv2d_grad<T, T, T, int16_t>(dev_ctx.x_context(),
                                             z->data<T>(),
                                             filter_z->data<T>(),
                                             conv_out_z_grad.data<T>(),
                                             z_grad_ptr,
                                             filter_z_grad_ptr,
                                             z_n,
                                             z_c,
                                             z_h,
                                             z_w,
                                             filter_z_shape[0],
                                             z_ksize,
                                             z_strides,
                                             paddings,
                                             dilations,
                                             group,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             is_nchw);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "resnet unit conv_grad");
    } else {
      auto conv_out_x_shape = phi::vectorize<int>(conv_out_x->dims());
      int bn_x_n = -1, bn_x_c = -1, bn_x_h = -1, bn_x_w = -1;
      fill_nchw(conv_out_x_shape, is_nchw, &bn_x_n, &bn_x_c, &bn_x_h, &bn_x_w);
      r = xpu::batch_norm_grad<T>(dev_ctx.x_context(),
                                  conv_out_x->data<T>(),
                                  relu_grad_ptr,
                                  conv_out_x_grad.data<T>(),
                                  bn_x_n,
                                  bn_x_c,
                                  bn_x_h,
                                  bn_x_w,
                                  scale_x->data<T>(),
                                  saved_mean_x->data<T>(),
                                  saved_invstd_x->data<T>(),
                                  scale_x_grad_ptr,
                                  bias_x_grad_ptr,
                                  is_nchw,
                                  nullptr,
                                  nullptr,
                                  eps);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "resnet unit bn_grad");

      Tensor *z_grad =
          fuse_add ? ctx.Output<Tensor>(framework::GradVarName("Z")) : nullptr;
      if (z_grad) {
        z_grad->mutable_data<T>(place);
        memory::Copy(place,
                     z_grad->data<T>(),
                     place,
                     relu_grad.data<T>(),
                     sizeof(T) * z_grad->numel());
      }
    }

    int x_n = -1, x_c = -1, x_h = -1, x_w = -1;
    fill_nchw(x_shape, is_nchw, &x_n, &x_c, &x_h, &x_w);
    std::vector<int> x_ksize = {filter_x_shape[2], filter_x_shape[3]};
    if (!is_nchw) {
      x_ksize[0] = filter_x_shape[1];
      x_ksize[1] = filter_x_shape[2];
    }
    r = xpu::conv2d_grad<T, T, T, int16_t>(dev_ctx.x_context(),
                                           x->data<T>(),
                                           filter_x->data<T>(),
                                           conv_out_x_grad.data<T>(),
                                           x_grad_ptr,
                                           filter_x_grad_ptr,
                                           x_n,
                                           x_c,
                                           x_h,
                                           x_w,
                                           filter_x_shape[0],
                                           x_ksize,
                                           x_strides,
                                           paddings,
                                           dilations,
                                           group,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           is_nchw);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "resnet unit conv_grad");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_XPU_KERNEL(resnet_unit, ops::ResNetUnitXPUKernel<float>);
REGISTER_OP_XPU_KERNEL(resnet_unit_grad, ops::ResNetUnitGradXPUKernel<float>);
