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
#include "paddle/phi/common/float16.h"

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
class ResNetUnitXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto place = ctx.GetPlace();
    PADDLE_ENFORCE_EQ(place.GetType() == phi::AllocationType::XPU,
                      true,
                      phi::errors::PreconditionNotMet("It must use XPUPlace."));

    bool is_nchw = (ctx.Attr<std::string>("data_format") == "NCHW");
    // input x
    const phi::DenseTensor *input_x = ctx.Input<phi::DenseTensor>("X");
    const phi::DenseTensor *filter_x = ctx.Input<phi::DenseTensor>("FilterX");
    const phi::DenseTensor *scale_x = ctx.Input<phi::DenseTensor>("ScaleX");
    const phi::DenseTensor *bias_x = ctx.Input<phi::DenseTensor>("BiasX");
    const phi::DenseTensor *maxptr_x = ctx.Input<phi::DenseTensor>("MaxPtrX");

    // output x
    phi::DenseTensor *conv_out_x = ctx.Output<phi::DenseTensor>("ConvX");
    phi::DenseTensor *saved_mean_x = ctx.Output<phi::DenseTensor>("SavedMeanX");
    phi::DenseTensor *saved_invstd_x =
        ctx.Output<phi::DenseTensor>("SavedInvstdX");
    phi::DenseTensor *running_mean_x =
        ctx.Output<phi::DenseTensor>("RunningMeanX");
    phi::DenseTensor *running_var_x =
        ctx.Output<phi::DenseTensor>("RunningVarX");

    phi::DenseTensor *output = ctx.Output<phi::DenseTensor>("Y");

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
    auto &dev_ctx = ctx.template device_context<phi::XPUContext>();

    std::vector<const XPUType *> x_list = {
        reinterpret_cast<const XPUType *>(input_x->data<T>())};
    std::vector<const XPUType *> w_list = {
        reinterpret_cast<const XPUType *>(filter_x->data<T>())};
    std::vector<XPUType *> conv_y_list = {
        reinterpret_cast<XPUType *>(conv_out_x->data<T>())};

    std::vector<std::vector<int>> x_shape_list = {
        common::vectorize<int>(input_x->dims())};

    auto filter_x_shape = common::vectorize<int>(filter_x->dims());
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
    std::vector<float *> batch_mean_list = {saved_mean_x->data<float>()};
    std::vector<float *> batch_invstd_list = {saved_invstd_x->data<float>()};
    std::vector<float *> global_mean_list = {running_mean_x->data<float>()};
    std::vector<float *> global_var_list = {running_var_x->data<float>()};

    std::vector<const float *> x_maxlist;
    if (maxptr_x) {
      const float *tmp = reinterpret_cast<const float *>(maxptr_x->data<int>());
      x_maxlist.push_back(tmp);
    } else {
      x_maxlist.push_back(nullptr);
    }
    std::vector<const float *> w_maxlist = {nullptr};
    if (has_shortcut) {
      // input z
      const phi::DenseTensor *input_z = ctx.Input<phi::DenseTensor>("Z");
      const phi::DenseTensor *filter_z = ctx.Input<phi::DenseTensor>("FilterZ");
      const phi::DenseTensor *scale_z = ctx.Input<phi::DenseTensor>("ScaleZ");
      const phi::DenseTensor *bias_z = ctx.Input<phi::DenseTensor>("BiasZ");
      const phi::DenseTensor *maxptr_z = ctx.Input<phi::DenseTensor>("MaxPtrZ");

      phi::DenseTensor *conv_out_z = ctx.Output<phi::DenseTensor>("ConvZ");
      phi::DenseTensor *saved_mean_z =
          ctx.Output<phi::DenseTensor>("SavedMeanZ");
      phi::DenseTensor *saved_invstd_z =
          ctx.Output<phi::DenseTensor>("SavedInvstdZ");
      phi::DenseTensor *running_mean_z =
          ctx.Output<phi::DenseTensor>("RunningMeanZ");
      phi::DenseTensor *running_var_z =
          ctx.Output<phi::DenseTensor>("RunningVarZ");

      x_list.push_back(reinterpret_cast<const XPUType *>(input_z->data<T>()));
      w_list.push_back(reinterpret_cast<const XPUType *>(filter_z->data<T>()));
      conv_y_list.push_back(reinterpret_cast<XPUType *>(conv_out_z->data<T>()));

      x_shape_list.push_back(common::vectorize<int>(input_z->dims()));

      auto filter_z_shape = common::vectorize<int>(filter_z->dims());
      std::vector<int> ksize_z = {filter_z_shape[2], filter_z_shape[3]};
      if (!is_nchw) {
        ksize_z[0] = filter_z_shape[1];
        ksize_z[1] = filter_z_shape[2];
      }
      ksize_list.push_back(ksize_z);
      stride_list.push_back({stride_z, stride_z});
      scale_list.push_back(scale_z->data<float>());
      bias_list.push_back(bias_z->data<float>());
      batch_mean_list.push_back(saved_mean_z->data<float>());
      batch_invstd_list.push_back(saved_invstd_z->data<float>());
      global_mean_list.push_back(running_mean_z->data<float>());
      global_var_list.push_back(running_var_z->data<float>());
      {
        const float *tmp =
            reinterpret_cast<const float *>(maxptr_z->data<int>());
        x_maxlist.push_back(tmp);
      }
      w_maxlist.push_back(nullptr);
    } else {
      if (fuse_add) {
        const phi::DenseTensor *input_z = ctx.Input<phi::DenseTensor>("Z");
        const phi::DenseTensor *maxptr_z =
            ctx.Input<phi::DenseTensor>("MaxPtrZ");
        auto input_z_shape = common::vectorize<int>(input_z->dims());
        x_list.push_back(reinterpret_cast<const XPUType *>(input_z->data<T>()));
        x_shape_list.push_back(input_z_shape);
        if (maxptr_z) {
          const float *tmp =
              reinterpret_cast<const float *>(maxptr_z->data<int>());
          x_maxlist.push_back(tmp);
        } else {
          x_maxlist.push_back(nullptr);
        }
      }
    }

    phi::DenseTensor *bitmask = ctx.Output<phi::DenseTensor>("BitMask");
    size_t bitmask_size = 0;
    if (std::getenv("XPU_PADDLE_FC_LOCAL_INT16") != nullptr) {
      bitmask_size = xpu::resnet_unit_fusion_get_reserve_space_size<XPUType,
                                                                    XPUType,
                                                                    XPUType,
                                                                    float>(
          dev_ctx.x_context(),
          x_shape_list,
          filter_x_shape[0],
          ksize_list,
          stride_list,
          paddings,
          dilations,
          group,
          x_maxlist,
          w_maxlist,
          xpu::Activation_t::RELU,
          is_nchw,
          has_shortcut,
          fuse_add);
    } else {
      bitmask_size = xpu::resnet_unit_fusion_get_reserve_space_size<XPUType,
                                                                    XPUType,
                                                                    XPUType,
                                                                    int16_t>(
          dev_ctx.x_context(),
          x_shape_list,
          filter_x_shape[0],
          ksize_list,
          stride_list,
          paddings,
          dilations,
          group,
          x_maxlist,
          w_maxlist,
          xpu::Activation_t::RELU,
          is_nchw,
          has_shortcut,
          fuse_add);
    }

    int64_t aligned_bitmask_size = (bitmask_size + 3) / 4;
    bitmask->Resize(
        phi::make_ddim({static_cast<int64_t>(aligned_bitmask_size)}));
    auto *bitmask_ptr = bitmask->data<int>();
    int r = 0;
    if (std::getenv("XPU_PADDLE_FC_LOCAL_INT16") != nullptr) {
      r = xpu::resnet_unit_fusion<XPUType, XPUType, XPUType, float>(
          dev_ctx.x_context(),
          x_list,
          w_list,
          conv_y_list,
          reinterpret_cast<XPUType *>(output->data<T>()),
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
          is_train,
          bitmask_ptr);
    } else {
      r = xpu::resnet_unit_fusion<XPUType, XPUType, XPUType, int16_t>(
          dev_ctx.x_context(),
          x_list,
          w_list,
          conv_y_list,
          reinterpret_cast<XPUType *>(output->data<T>()),
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
          is_train,
          bitmask_ptr);
    }
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "resnet_unit_fusion");
  }
};

template <typename T, typename DeviceContext>
class ResNetUnitGradXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto place = ctx.GetPlace();
    PADDLE_ENFORCE_EQ(place.GetType() == phi::AllocationType::XPU,
                      true,
                      phi::errors::PreconditionNotMet("It must use XPUPlace."));

    bool is_nchw = (ctx.Attr<std::string>("data_format") == "NCHW");
    const phi::DenseTensor *y_grad =
        ctx.Input<phi::DenseTensor>(framework::GradVarName("Y"));
    const phi::DenseTensor *x = ctx.Input<phi::DenseTensor>("X");
    const phi::DenseTensor *filter_x = ctx.Input<phi::DenseTensor>("FilterX");
    const phi::DenseTensor *scale_x = ctx.Input<phi::DenseTensor>("ScaleX");
    const phi::DenseTensor *saved_mean_x =
        ctx.Input<phi::DenseTensor>("SavedMeanX");
    const phi::DenseTensor *saved_invstd_x =
        ctx.Input<phi::DenseTensor>("SavedInvstdX");
    const phi::DenseTensor *conv_out_x = ctx.Input<phi::DenseTensor>("ConvX");
    const phi::DenseTensor *output = ctx.Input<phi::DenseTensor>("Y");
    const phi::DenseTensor *maxptr_x = ctx.Input<phi::DenseTensor>("MaxPtrX");

    phi::DenseTensor *x_grad = nullptr;
    int has_dx = ctx.Attr<bool>("has_dx");
    if (has_dx) {
      x_grad = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    }
    phi::DenseTensor *filter_x_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("FilterX"));
    phi::DenseTensor *scale_x_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("ScaleX"));
    phi::DenseTensor *bias_x_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("BiasX"));

    int padding = ctx.Attr<int>("padding");
    int stride = ctx.Attr<int>("stride");
    int stride_z = ctx.Attr<int>("stride_z");
    int dilation = ctx.Attr<int>("dilation");
    int group = ctx.Attr<int>("group");
    float eps = ctx.Attr<float>("epsilon");
    bool has_shortcut = ctx.Attr<bool>("has_shortcut");
    bool fuse_add = ctx.Attr<bool>("fuse_add");
    std::string act_type = ctx.Attr<std::string>("act_type");

    auto &dev_ctx = ctx.template device_context<phi::XPUContext>();

    std::vector<const XPUType *> x_list = {
        reinterpret_cast<const XPUType *>(x->data<T>())};
    std::vector<const XPUType *> w_list = {
        reinterpret_cast<const XPUType *>(filter_x->data<T>())};
    std::vector<const XPUType *> conv_y_list = {
        reinterpret_cast<const XPUType *>(conv_out_x->data<T>())};
    std::vector<XPUType *> dx_list = {nullptr};
    if (has_dx) {
      dx_list[0] = reinterpret_cast<XPUType *>(x_grad->data<T>());
    }
    std::vector<XPUType *> dw_list = {
        reinterpret_cast<XPUType *>(filter_x_grad->data<T>())};

    std::vector<std::vector<int>> x_shape_list = {
        common::vectorize<int>(x->dims())};

    auto filter_x_shape = common::vectorize<int>(filter_x->dims());
    std::vector<int> x_ksize = {filter_x_shape[2], filter_x_shape[3]};
    if (!is_nchw) {
      x_ksize[0] = filter_x_shape[1];
      x_ksize[1] = filter_x_shape[2];
    }
    std::vector<std::vector<int>> ksize_list = {x_ksize};
    std::vector<std::vector<int>> stride_list = {{stride, stride}};
    std::vector<int> paddings = {padding, padding};
    std::vector<int> dilations = {dilation, dilation};

    std::vector<const float *> x_maxlist;
    if (maxptr_x) {
      const float *tmp = reinterpret_cast<const float *>(maxptr_x->data<int>());
      x_maxlist.push_back(tmp);
    } else {
      x_maxlist.push_back(nullptr);
    }
    std::vector<const float *> w_maxlist = {nullptr};

    std::vector<const float *> scale_list = {scale_x->data<float>()};
    std::vector<const float *> batch_mean_list = {saved_mean_x->data<float>()};
    std::vector<const float *> batch_invstd_list = {
        saved_invstd_x->data<float>()};
    std::vector<float *> dscale_list = {scale_x_grad->data<float>()};
    std::vector<float *> dbias_list = {bias_x_grad->data<float>()};

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
      const phi::DenseTensor *z = ctx.Input<phi::DenseTensor>("Z");
      const phi::DenseTensor *filter_z = ctx.Input<phi::DenseTensor>("FilterZ");
      const phi::DenseTensor *scale_z = ctx.Input<phi::DenseTensor>("ScaleZ");
      const phi::DenseTensor *saved_mean_z =
          ctx.Input<phi::DenseTensor>("SavedMeanZ");
      const phi::DenseTensor *saved_invstd_z =
          ctx.Input<phi::DenseTensor>("SavedInvstdZ");
      const phi::DenseTensor *conv_out_z = ctx.Input<phi::DenseTensor>("ConvZ");
      const phi::DenseTensor *maxptr_z = ctx.Input<phi::DenseTensor>("MaxPtrZ");

      phi::DenseTensor *z_grad =
          ctx.Output<phi::DenseTensor>(framework::GradVarName("Z"));
      phi::DenseTensor *filter_z_grad =
          ctx.Output<phi::DenseTensor>(framework::GradVarName("FilterZ"));
      phi::DenseTensor *scale_z_grad =
          ctx.Output<phi::DenseTensor>(framework::GradVarName("ScaleZ"));
      phi::DenseTensor *bias_z_grad =
          ctx.Output<phi::DenseTensor>(framework::GradVarName("BiasZ"));
      x_list.push_back(reinterpret_cast<const XPUType *>(z->data<T>()));
      w_list.push_back(reinterpret_cast<const XPUType *>(filter_z->data<T>()));
      conv_y_list.push_back(
          reinterpret_cast<const XPUType *>(conv_out_z->data<T>()));
      dx_list.push_back(reinterpret_cast<XPUType *>(z_grad->data<T>()));
      dw_list.push_back(reinterpret_cast<XPUType *>(filter_z_grad->data<T>()));
      x_shape_list.push_back(common::vectorize<int>(z->dims()));

      auto filter_z_shape = common::vectorize<int>(filter_z->dims());
      std::vector<int> ksize_z = {filter_z_shape[2], filter_z_shape[3]};
      if (!is_nchw) {
        ksize_z[0] = filter_z_shape[1];
        ksize_z[1] = filter_z_shape[2];
      }
      ksize_list.push_back(ksize_z);
      stride_list.push_back({stride_z, stride_z});
      const float *tmp = reinterpret_cast<const float *>(maxptr_z->data<int>());
      x_maxlist.push_back(tmp);
      w_maxlist.push_back(nullptr);

      scale_list.push_back(scale_z->data<float>());
      batch_mean_list.push_back(saved_mean_z->data<float>());
      batch_invstd_list.push_back(saved_invstd_z->data<float>());
      dscale_list.push_back(scale_z_grad->data<float>());
      dbias_list.push_back(bias_z_grad->data<float>());
    } else {
      if (fuse_add) {
        auto z_grad = ctx.Output<phi::DenseTensor>(framework::GradVarName("Z"));
        dx_list.push_back(reinterpret_cast<XPUType *>(z_grad->data<T>()));
      }
    }

    const phi::DenseTensor *bitmask = ctx.Input<phi::DenseTensor>("BitMask");
    int r = 0;
    if (std::getenv("XPU_PADDLE_FC_LOCAL_INT16") != nullptr) {
      r = xpu::resnet_unit_grad_fusion<XPUType, XPUType, XPUType, float>(
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
          fuse_add,
          reinterpret_cast<void *>(const_cast<int *>(bitmask->data<int>())));
    } else {
      r = xpu::resnet_unit_grad_fusion<XPUType, XPUType, XPUType, int16_t>(
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
          fuse_add,
          reinterpret_cast<void *>(const_cast<int *>(bitmask->data<int>())));
    }
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "resnet_unit_grad_fusion");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

PD_REGISTER_STRUCT_KERNEL(resnet_unit,
                          XPU,
                          ALL_LAYOUT,
                          ops::ResNetUnitXPUKernel,
                          phi::dtype::float16,
                          float) {}
PD_REGISTER_STRUCT_KERNEL(resnet_unit_grad,
                          XPU,
                          ALL_LAYOUT,
                          ops::ResNetUnitGradXPUKernel,
                          phi::dtype::float16,
                          float) {}
