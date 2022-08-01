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

template <typename T>
class ResNetUnitXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto place = ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        platform::is_xpu_place(place),
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
    auto &dev_ctx = ctx.template device_context<platform::XPUDeviceContext>();

    std::vector<const XPUType *> x_list = {
        reinterpret_cast<const XPUType *>(input_x->data<T>())};
    std::vector<const XPUType *> w_list = {
        reinterpret_cast<const XPUType *>(filter_x->data<T>())};
    std::vector<XPUType *> conv_y_list = {
        reinterpret_cast<XPUType *>(conv_out_x->mutable_data<T>(place))};

    std::vector<std::vector<int>> x_shape_list = {
        phi::vectorize<int>(input_x->dims())};

    auto filter_x_shape = phi::vectorize<int>(filter_x->dims());
    std::vector<int> ksize = {filter_x_shape[2], filter_x_shape[3]};
    if (!is_nchw) {
      ksize[0] = filter_x_shape[1];
      ksize[1] = filter_x_shape[2];
    }
    std::vector<int> strides = {stride, stride};
    std::vector<std::vector<int>> ksize_list = {ksize};
    std::vector<std::vector<int>> stride_list = {strides};
    std::vector<int> paddings = {padding, padding};
    std::vector<int> dilations = {dilation, dilation};
    std::vector<const float *> scale_list = {scale_x->data<float>()};
    std::vector<const float *> bias_list = {bias_x->data<float>()};
    std::vector<float *> batch_mean_list = {
        saved_mean_x->mutable_data<float>(place)};
    std::vector<float *> batch_invstd_list = {
        saved_invstd_x->mutable_data<float>(place)};
    std::vector<float *> global_mean_list = {
        running_mean_x->mutable_data<float>(place)};
    std::vector<float *> global_var_list = {
        running_var_x->mutable_data<float>(place)};

    std::vector<const float *> x_maxlist = {nullptr};
    std::vector<const float *> w_maxlist = {nullptr};
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

      x_list.push_back(reinterpret_cast<const XPUType *>(input_z->data<T>()));
      w_list.push_back(reinterpret_cast<const XPUType *>(filter_z->data<T>()));
      conv_y_list.push_back(
          reinterpret_cast<XPUType *>(conv_out_z->mutable_data<T>(place)));

      x_shape_list.push_back(phi::vectorize<int>(input_z->dims()));

      auto filter_z_shape = phi::vectorize<int>(filter_z->dims());
      std::vector<int> ksize_z = {filter_z_shape[2], filter_z_shape[3]};
      if (!is_nchw) {
        ksize_z[0] = filter_z_shape[1];
        ksize_z[1] = filter_z_shape[2];
      }
      ksize_list.push_back(ksize_z);
      stride_list.push_back({stride_z, stride_z});
      scale_list.push_back(scale_z->data<float>());
      bias_list.push_back(bias_z->data<float>());
      batch_mean_list.push_back(saved_mean_z->mutable_data<float>(place));
      batch_invstd_list.push_back(saved_invstd_z->mutable_data<float>(place));
      global_mean_list.push_back(running_mean_z->mutable_data<float>(place));
      global_var_list.push_back(running_var_z->mutable_data<float>(place));
      x_maxlist.push_back(nullptr);
      w_maxlist.push_back(nullptr);
    } else {
      if (fuse_add) {
        const Tensor *input_z = ctx.Input<Tensor>("Z");
        auto input_z_shape = phi::vectorize<int>(input_z->dims());
        x_list.push_back(reinterpret_cast<const XPUType *>(input_z->data<T>()));
        x_shape_list.push_back(input_z_shape);
        x_maxlist.push_back(nullptr);
      }
    }
    int r = xpu::resnet_unit_fusion<XPUType, XPUType, XPUType, int16_t>(
        dev_ctx.x_context(),
        x_list,
        w_list,
        conv_y_list,
        reinterpret_cast<XPUType *>(output->mutable_data<T>(place)),
        x_shape_list,
        filter_x_shape[0],
        ksize_list,
        stride_list,
        paddings,
        dilations,
        group,
        eps,
        momentum,
        x_maxlist,
        w_maxlist,
        scale_list,
        bias_list,
        batch_mean_list,
        batch_invstd_list,
        global_mean_list,
        global_var_list,
        xpu::Activation_t::RELU,
        is_nchw,
        has_shortcut,
        fuse_add,
        is_train);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "resnet_unit_fusion");
  }
};

template <typename T>
class ResNetUnitGradXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

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

    int padding = ctx.Attr<int>("padding");
    int stride = ctx.Attr<int>("stride");
    int stride_z = ctx.Attr<int>("stride_z");
    int dilation = ctx.Attr<int>("dilation");
    int group = ctx.Attr<int>("group");
    float eps = ctx.Attr<float>("epsilon");
    bool has_shortcut = ctx.Attr<bool>("has_shortcut");
    bool fuse_add = ctx.Attr<bool>("fuse_add");
    std::string act_type = ctx.Attr<std::string>("act_type");

    auto &dev_ctx = ctx.template device_context<platform::XPUDeviceContext>();

    std::vector<const XPUType *> x_list = {
        reinterpret_cast<const XPUType *>(x->data<T>())};
    std::vector<const XPUType *> w_list = {
        reinterpret_cast<const XPUType *>(filter_x->data<T>())};
    std::vector<const XPUType *> conv_y_list = {
        reinterpret_cast<const XPUType *>(conv_out_x->data<T>())};
    std::vector<XPUType *> dx_list = {
        reinterpret_cast<XPUType *>(x_grad->mutable_data<T>(place))};
    std::vector<XPUType *> dw_list = {
        reinterpret_cast<XPUType *>(filter_x_grad->mutable_data<T>(place))};

    std::vector<std::vector<int>> x_shape_list = {
        phi::vectorize<int>(x->dims())};

    auto filter_x_shape = phi::vectorize<int>(filter_x->dims());
    std::vector<int> x_ksize = {filter_x_shape[2], filter_x_shape[3]};
    if (!is_nchw) {
      x_ksize[0] = filter_x_shape[1];
      x_ksize[1] = filter_x_shape[2];
    }
    std::vector<std::vector<int>> ksize_list = {x_ksize};
    std::vector<std::vector<int>> stride_list = {{stride, stride}};
    std::vector<int> paddings = {padding, padding};
    std::vector<int> dilations = {dilation, dilation};

    std::vector<const float *> x_maxlist = {nullptr};
    std::vector<const float *> w_maxlist = {nullptr};

    std::vector<const float *> scale_list = {scale_x->data<float>()};
    std::vector<const float *> batch_mean_list = {saved_mean_x->data<float>()};
    std::vector<const float *> batch_invstd_list = {
        saved_invstd_x->data<float>()};
    std::vector<float *> dscale_list = {
        scale_x_grad->mutable_data<float>(place)};
    std::vector<float *> dbias_list = {bias_x_grad->mutable_data<float>(place)};

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
      x_list.push_back(reinterpret_cast<const XPUType *>(z->data<T>()));
      w_list.push_back(reinterpret_cast<const XPUType *>(filter_z->data<T>()));
      conv_y_list.push_back(
          reinterpret_cast<const XPUType *>(conv_out_z->data<T>()));
      dx_list.push_back(
          reinterpret_cast<XPUType *>(z_grad->mutable_data<T>(place)));
      dw_list.push_back(
          reinterpret_cast<XPUType *>(filter_z_grad->mutable_data<T>(place)));
      x_shape_list.push_back(phi::vectorize<int>(z->dims()));

      auto filter_z_shape = phi::vectorize<int>(filter_z->dims());
      std::vector<int> ksize_z = {filter_z_shape[2], filter_z_shape[3]};
      if (!is_nchw) {
        ksize_z[0] = filter_z_shape[1];
        ksize_z[1] = filter_z_shape[2];
      }
      ksize_list.push_back(ksize_z);
      stride_list.push_back({stride_z, stride_z});
      x_maxlist.push_back(nullptr);
      w_maxlist.push_back(nullptr);

      scale_list.push_back(scale_z->data<float>());
      batch_mean_list.push_back(saved_mean_z->data<float>());
      batch_invstd_list.push_back(saved_invstd_z->data<float>());
      dscale_list.push_back(scale_z_grad->mutable_data<float>(place));
      dbias_list.push_back(bias_z_grad->mutable_data<float>(place));
    } else {
      if (fuse_add) {
        auto z_grad = ctx.Output<Tensor>(framework::GradVarName("Z"));
        dx_list.push_back(
            reinterpret_cast<XPUType *>(z_grad->mutable_data<T>(place)));
      }
    }

    int r = xpu::resnet_unit_grad_fusion<XPUType, XPUType, XPUType, int16_t>(
        dev_ctx.x_context(),
        x_list,
        w_list,
        reinterpret_cast<const XPUType *>(y_grad->data<T>()),
        reinterpret_cast<const XPUType *>(output->data<T>()),
        conv_y_list,
        dx_list,
        dw_list,
        x_shape_list,
        filter_x_shape[0],
        ksize_list,
        stride_list,
        paddings,
        dilations,
        group,
        x_maxlist,
        w_maxlist,
        scale_list,
        batch_mean_list,
        batch_invstd_list,
        dscale_list,
        dbias_list,
        xpu::Activation_t::RELU,
        eps,
        is_nchw,
        has_shortcut,
        fuse_add);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "resnet_unit_grad_fusion");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_XPU_KERNEL(resnet_unit,
                       ops::ResNetUnitXPUKernel<plat::float16>,
                       ops::ResNetUnitXPUKernel<float>);
REGISTER_OP_XPU_KERNEL(resnet_unit_grad,
                       ops::ResNetUnitGradXPUKernel<plat::float16>,
                       ops::ResNetUnitGradXPUKernel<float>);
