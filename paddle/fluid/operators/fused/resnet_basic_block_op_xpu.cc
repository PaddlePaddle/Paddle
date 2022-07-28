// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class ResnetBasicBlockAttr {
 public:
  explicit ResnetBasicBlockAttr(const framework::ExecutionContext& ctx) {
    padding1 = ctx.Attr<int>("padding1");
    padding2 = ctx.Attr<int>("padding2");
    padding3 = ctx.Attr<int>("padding3");
    stride1 = ctx.Attr<int>("stride1");
    stride2 = ctx.Attr<int>("stride2");
    stride3 = ctx.Attr<int>("stride3");
    dilation1 = ctx.Attr<int>("dilation1");
    dilation2 = ctx.Attr<int>("dilation2");
    dilation3 = ctx.Attr<int>("dilation3");
    group = ctx.Attr<int>("group");

    eps = static_cast<double>(ctx.Attr<float>("epsilon"));
    momentum = static_cast<double>(ctx.Attr<float>("momentum"));
    has_shortcut = ctx.Attr<bool>("has_shortcut");
    find_max = ctx.Attr<bool>("find_conv_input_max");

    const auto is_test = ctx.Attr<bool>("is_test");
    const auto use_global_stats = ctx.Attr<bool>("use_global_stats");
    const auto trainable_stats = ctx.Attr<bool>("trainable_statistics");
    bool test_mode = is_test && (!trainable_stats);
    global_stats = test_mode || use_global_stats;

    // init shape
    auto input1 = ctx.Input<Tensor>("X");
    auto filter1 = ctx.Input<Tensor>("Filter1");
    auto conv1_out = ctx.Output<Tensor>("Conv1");
    auto filter2 = ctx.Input<Tensor>("Filter2");
    auto conv2_out = ctx.Output<Tensor>("Conv2");
    conv1_input_shape = phi::vectorize<int>(input1->dims());
    conv1_output_shape = phi::vectorize<int>(conv1_out->dims());
    conv1_filter_shape = phi::vectorize<int>(filter1->dims());
    conv1_filter_numel = filter1->numel();
    conv1_input_numel = input1->numel();
    conv1_output_numel = conv1_out->numel();

    conv2_input_shape = phi::vectorize<int>(conv1_out->dims());
    conv2_output_shape = phi::vectorize<int>(conv2_out->dims());
    conv2_filter_shape = phi::vectorize<int>(filter2->dims());
    conv2_filter_numel = filter2->numel();
    conv2_input_numel = conv1_out->numel();
    conv2_output_numel = conv2_out->numel();

    if (has_shortcut) {
      auto filter3 = ctx.Input<Tensor>("Filter3");
      auto conv3_out = ctx.Output<Tensor>("Conv3");
      conv3_input_shape = phi::vectorize<int>(input1->dims());
      conv3_output_shape = phi::vectorize<int>(conv3_out->dims());
      conv3_filter_shape = phi::vectorize<int>(filter3->dims());
      conv3_filter_numel = filter3->numel();
      conv3_input_numel = input1->numel();
      conv3_output_numel = conv3_out->numel();
    }
  }

  int padding1;
  int padding2;
  int padding3;
  int stride1;
  int stride2;
  int stride3;
  int dilation1;
  int dilation2;
  int dilation3;
  int group;

  double eps;
  double momentum;

  bool has_shortcut;
  bool find_max;
  bool global_stats;

  std::vector<int> conv1_input_shape;
  std::vector<int> conv1_output_shape;
  std::vector<int> conv1_filter_shape;
  std::vector<int> conv2_input_shape;
  std::vector<int> conv2_output_shape;
  std::vector<int> conv2_filter_shape;
  std::vector<int> conv3_input_shape;
  std::vector<int> conv3_output_shape;
  std::vector<int> conv3_filter_shape;

  int conv1_filter_numel;
  int conv2_filter_numel;
  int conv3_filter_numel;
  int conv1_input_numel;
  int conv2_input_numel;
  int conv3_input_numel;
  int conv1_output_numel;
  int conv2_output_numel;
  int conv3_output_numel;
};

class ResnetBasicBlockGradAttr {
 public:
  explicit ResnetBasicBlockGradAttr(const framework::ExecutionContext& ctx) {
    padding1 = ctx.Attr<int>("padding1");
    padding2 = ctx.Attr<int>("padding2");
    padding3 = ctx.Attr<int>("padding3");
    stride1 = ctx.Attr<int>("stride1");
    stride2 = ctx.Attr<int>("stride2");
    stride3 = ctx.Attr<int>("stride3");
    dilation1 = ctx.Attr<int>("dilation1");
    dilation2 = ctx.Attr<int>("dilation2");
    dilation3 = ctx.Attr<int>("dilation3");
    group = ctx.Attr<int>("group");

    has_shortcut = ctx.Attr<bool>("has_shortcut");
    find_max = ctx.Attr<bool>("find_conv_input_max");

    // init shape
    auto input1 = ctx.Input<Tensor>("X");
    auto filter1 = ctx.Input<Tensor>("Filter1");
    auto conv1_out = ctx.Input<Tensor>("Conv1");
    auto filter2 = ctx.Input<Tensor>("Filter2");
    auto conv2_out = ctx.Input<Tensor>("Conv2");
    conv1_input_shape = phi::vectorize<int>(input1->dims());
    conv1_output_shape = phi::vectorize<int>(conv1_out->dims());
    conv1_filter_shape = phi::vectorize<int>(filter1->dims());
    conv1_filter_numel = filter1->numel();
    conv1_input_numel = input1->numel();
    conv1_output_numel = conv1_out->numel();

    conv2_input_shape = phi::vectorize<int>(conv1_out->dims());
    conv2_output_shape = phi::vectorize<int>(conv2_out->dims());
    conv2_filter_shape = phi::vectorize<int>(filter2->dims());
    conv2_filter_numel = filter2->numel();
    conv2_input_numel = conv1_out->numel();
    conv2_output_numel = conv2_out->numel();

    if (has_shortcut) {
      auto filter3 = ctx.Input<Tensor>("Filter3");
      auto conv3_out = ctx.Input<Tensor>("Conv3");
      conv3_input_shape = phi::vectorize<int>(input1->dims());
      conv3_output_shape = phi::vectorize<int>(conv3_out->dims());
      conv3_filter_shape = phi::vectorize<int>(filter3->dims());
      conv3_filter_numel = filter3->numel();
      conv3_input_numel = input1->numel();
      conv3_output_numel = conv3_out->numel();
    }
  }

  int padding1;
  int padding2;
  int padding3;
  int stride1;
  int stride2;
  int stride3;
  int dilation1;
  int dilation2;
  int dilation3;
  int group;

  bool has_shortcut;
  bool find_max;

  std::vector<int> conv1_input_shape;
  std::vector<int> conv1_output_shape;
  std::vector<int> conv1_filter_shape;
  std::vector<int> conv2_input_shape;
  std::vector<int> conv2_output_shape;
  std::vector<int> conv2_filter_shape;
  std::vector<int> conv3_input_shape;
  std::vector<int> conv3_output_shape;
  std::vector<int> conv3_filter_shape;

  int conv1_filter_numel;
  int conv2_filter_numel;
  int conv3_filter_numel;
  int conv1_input_numel;
  int conv2_input_numel;
  int conv3_input_numel;
  int conv1_output_numel;
  int conv2_output_numel;
  int conv3_output_numel;
};

template <typename T>
static inline void xpu_conv2d(xpu::Context* ctx,
                              const T* input_data,
                              const T* filter_data,
                              T* output_data,
                              float* input_max_data,
                              float* filter_max_data,
                              const std::vector<int>& input_shape,
                              const std::vector<int>& filter_shape,
                              int padding,
                              int stride,
                              int dilation,
                              int group) {
  std::vector<int> ksize{filter_shape[2], filter_shape[3]};
  std::vector<int> stride_vec{stride, stride};
  std::vector<int> dilation_vec{dilation, dilation};
  std::vector<int> padding_vec{padding, padding};
  int N = input_shape[0];
  int C = input_shape[1];
  int H = input_shape[2];
  int W = input_shape[3];

  int r = xpu::conv2d<T, T, T, int16_t>(ctx,
                                        input_data,
                                        filter_data,
                                        output_data,
                                        N,
                                        C,
                                        H,
                                        W,
                                        filter_shape[0],
                                        ksize,
                                        stride_vec,
                                        padding_vec,
                                        dilation_vec,
                                        group,
                                        input_max_data,
                                        filter_max_data,
                                        nullptr,
                                        true);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d");
}

template <typename T>
static inline void xpu_conv2d_grad(xpu::Context* ctx,
                                   const T* input_data,
                                   const T* filter_data,
                                   const T* output_grad_data,
                                   T* input_grad_data,
                                   T* filter_grad_data,
                                   const float* input_max_data,
                                   const float* filter_max_data,
                                   const std::vector<int>& input_shape,
                                   const std::vector<int>& filter_shape,
                                   int padding,
                                   int stride,
                                   int dilation,
                                   int group) {
  std::vector<int> ksize{filter_shape[2], filter_shape[3]};
  std::vector<int> stride_vec{stride, stride};
  std::vector<int> dilation_vec{dilation, dilation};
  std::vector<int> padding_vec{padding, padding};
  int N = input_shape[0];
  int C = input_shape[1];
  int H = input_shape[2];
  int W = input_shape[3];

  int r = xpu::conv2d_grad<T, T, T, int16_t>(ctx,
                                             input_data,
                                             filter_data,
                                             output_grad_data,
                                             input_grad_data,
                                             filter_grad_data,
                                             N,
                                             C,
                                             H,
                                             W,
                                             filter_shape[0],
                                             ksize,
                                             stride_vec,
                                             padding_vec,
                                             dilation_vec,
                                             group,
                                             input_max_data,
                                             filter_max_data,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             true);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d_grad");
}

template <typename T>
class ResNetBasicBlockXPUKernel : public framework::OpKernel<T> {
 public:
  using XPUT = typename XPUTypeTrait<T>::Type;

  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_xpu_place(ctx.GetPlace()),
        true,
        platform::errors::PreconditionNotMet("It must use XPUPlace."));

    // input
    const Tensor* x = ctx.Input<Tensor>("X");
    const Tensor* filter1 = ctx.Input<Tensor>("Filter1");
    const Tensor* scale1 = ctx.Input<Tensor>("Scale1");
    const Tensor* bias1 = ctx.Input<Tensor>("Bias1");
    const Tensor* filter2 = ctx.Input<Tensor>("Filter2");
    const Tensor* scale2 = ctx.Input<Tensor>("Scale2");
    const Tensor* bias2 = ctx.Input<Tensor>("Bias2");

    // output
    Tensor* conv1_output = ctx.Output<Tensor>("Conv1");
    Tensor* conv2_output = ctx.Output<Tensor>("Conv2");
    Tensor* conv2_input = ctx.Output<Tensor>("Conv2Input");
    Tensor* output = ctx.Output<Tensor>("Y");

    auto place = ctx.GetPlace();
    auto x_data = reinterpret_cast<const XPUT*>(x->data<T>());
    auto conv1_filter_data = reinterpret_cast<const XPUT*>(filter1->data<T>());
    auto conv2_filter_data = reinterpret_cast<const XPUT*>(filter2->data<T>());
    auto conv1_output_data =
        reinterpret_cast<XPUT*>(conv1_output->mutable_data<T>(place));
    auto conv2_input_data =
        reinterpret_cast<XPUT*>(conv2_input->mutable_data<T>(place));
    auto conv2_output_data =
        reinterpret_cast<XPUT*>(conv2_output->mutable_data<T>(place));
    auto scale1_data = scale1->data<float>();
    auto scale2_data = scale2->data<float>();
    auto bias1_data = bias1->data<float>();
    auto bias2_data = bias2->data<float>();
    auto output_data = reinterpret_cast<XPUT*>(output->mutable_data<T>(place));

    float* conv1_input_max_data = nullptr;
    float* conv1_filter_max_data = nullptr;
    float* conv2_input_max_data = nullptr;
    float* conv2_filter_max_data = nullptr;
    float* conv3_input_max_data = nullptr;
    float* conv3_filter_max_data = nullptr;

    ResnetBasicBlockAttr attr(ctx);

    // init find max
    if (attr.find_max) {
      Tensor* max_input1 = ctx.Output<Tensor>("MaxInput1");
      Tensor* max_filter1 = ctx.Output<Tensor>("MaxFilter1");
      conv1_input_max_data = max_input1->mutable_data<float>(place);
      conv1_filter_max_data = max_filter1->mutable_data<float>(place);

      Tensor* max_input2 = ctx.Output<Tensor>("MaxInput2");
      Tensor* max_filter2 = ctx.Output<Tensor>("MaxFilter2");
      conv2_input_max_data = max_input2->mutable_data<float>(place);
      conv2_filter_max_data = max_filter2->mutable_data<float>(place);

      if (attr.has_shortcut) {
        Tensor* max_input3 = ctx.Output<Tensor>("MaxInput3");
        Tensor* max_filter3 = ctx.Output<Tensor>("MaxFilter3");
        conv3_input_max_data = max_input3->mutable_data<float>(place);
        conv3_filter_max_data = max_filter3->mutable_data<float>(place);
      }
    }

    auto& dev_ctx = ctx.template device_context<platform::XPUDeviceContext>();
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    int r = XPU_SUCCESS;

    // 1. short
    const XPUT* z_out_data = nullptr;
    if (attr.has_shortcut) {
      Tensor* conv3_out = ctx.Output<Tensor>("Conv3");
      const Tensor* filter3 = ctx.Input<Tensor>("Filter3");
      auto conv3_filter_data =
          reinterpret_cast<const XPUT*>(filter3->data<T>());
      auto conv3_output_data =
          reinterpret_cast<XPUT*>(conv3_out->mutable_data<T>(place));

      XPUT* conv3_input_l3_data = nullptr;
      XPUT* conv3_filter_l3_data =
          RAII_GUARD.alloc_l3<XPUT>(attr.conv3_filter_numel);

      if (attr.find_max) {
        r = xpu::findmax_copy_fusion(dev_ctx.x_context(),
                                     x_data,
                                     conv3_input_max_data,
                                     conv3_input_l3_data,
                                     attr.conv3_input_numel);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "findmax_copy_fusion");

        r = xpu::findmax_copy_fusion(dev_ctx.x_context(),
                                     conv3_filter_data,
                                     conv3_filter_max_data,
                                     conv3_filter_l3_data,
                                     attr.conv3_filter_numel);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "findmax_copy_fusion");
      }

      xpu_conv2d(dev_ctx.x_context(),
                 conv3_input_l3_data != nullptr ? conv3_input_l3_data : x_data,
                 conv3_filter_l3_data,
                 conv3_output_data,
                 conv3_input_max_data,
                 conv3_filter_max_data,
                 attr.conv3_input_shape,
                 attr.conv3_filter_shape,
                 attr.padding3,
                 attr.stride3,
                 attr.dilation3,
                 attr.group);

      // bn3
      const Tensor* scale3 = ctx.Input<Tensor>("Scale3");
      const Tensor* bias3 = ctx.Input<Tensor>("Bias3");
      auto bias3_data = bias3->data<float>();
      auto scale3_data = scale3->data<float>();

      auto bn3_output_data = RAII_GUARD.alloc<XPUT>(attr.conv3_output_numel);
      PADDLE_ENFORCE_XDNN_NOT_NULL(bn3_output_data);

      if (!attr.global_stats) {
        Tensor* saved_mean3 = ctx.Output<Tensor>("SavedMean3");
        Tensor* saved_invstd3 = ctx.Output<Tensor>("SavedInvstd3");
        Tensor* running_mean3 = ctx.Output<Tensor>("Mean3Out");
        Tensor* running_var3 = ctx.Output<Tensor>("Var3Out");

        auto saved_mean3_data = saved_mean3->mutable_data<float>(place);
        auto saved_invstd3_data = saved_invstd3->mutable_data<float>(place);
        auto running_mean3_data = running_mean3->mutable_data<float>(place);
        auto running_var3_data = running_var3->mutable_data<float>(place);

        r = xpu::batch_norm_fusion<XPUT>(dev_ctx.x_context(),
                                         conv3_output_data,
                                         bn3_output_data,
                                         attr.conv3_output_shape[0],
                                         attr.conv3_output_shape[1],
                                         attr.conv3_output_shape[3],
                                         attr.conv3_output_shape[3],
                                         attr.eps,
                                         attr.momentum,
                                         scale3_data,
                                         bias3_data,
                                         saved_mean3_data,
                                         saved_invstd3_data,
                                         running_mean3_data,
                                         running_var3_data,
                                         true,
                                         nullptr,
                                         xpu::Activation_t::LINEAR,
                                         nullptr,
                                         0);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "batch_norm_fusion");
      } else {
        const auto* mean3 = ctx.Input<Tensor>("Mean3");
        const auto* var3 = ctx.Input<Tensor>("Var3");
        const auto* mean3_data = mean3->data<float>();
        const auto* variance3_data = var3->data<float>();
        r = xpu::batch_norm_infer<XPUT>(dev_ctx.x_context(),
                                        conv3_output_data,
                                        bn3_output_data,
                                        attr.conv3_output_shape[0],
                                        attr.conv3_output_shape[1],
                                        attr.conv3_output_shape[2],
                                        attr.conv3_output_shape[3],
                                        attr.eps,
                                        scale3_data,
                                        bias3_data,
                                        mean3_data,
                                        variance3_data,
                                        true);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "batch_norm_infer");
      }
      z_out_data = reinterpret_cast<const XPUT*>(bn3_output_data);
    } else {
      z_out_data = x_data;
    }

    // 2. conv1
    XPUT* conv1_input_l3_data = nullptr;
    XPUT* conv1_filter_l3_data =
        RAII_GUARD.alloc_l3<XPUT>(attr.conv1_filter_numel);
    if (attr.find_max) {
      r = xpu::findmax_copy_fusion(dev_ctx.x_context(),
                                   x_data,
                                   conv1_input_max_data,
                                   conv1_input_l3_data,
                                   attr.conv1_input_numel);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "findmax_copy_fusion");

      r = xpu::findmax_copy_fusion(dev_ctx.x_context(),
                                   conv1_filter_data,
                                   conv1_filter_max_data,
                                   conv1_filter_l3_data,
                                   attr.conv1_filter_numel);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "findmax_copy_fusion");
    }
    xpu_conv2d(dev_ctx.x_context(),
               conv1_input_l3_data != nullptr ? conv1_input_l3_data : x_data,
               conv1_filter_l3_data,
               conv1_output_data,
               conv1_input_max_data,
               conv1_filter_max_data,
               attr.conv1_input_shape,
               attr.conv1_filter_shape,
               attr.padding1,
               attr.stride1,
               attr.dilation1,
               attr.group);

    // 3. bn1 + relu
    if (!attr.global_stats) {
      Tensor* saved_mean1 = ctx.Output<Tensor>("SavedMean1");
      Tensor* saved_invstd1 = ctx.Output<Tensor>("SavedInvstd1");
      Tensor* running_mean1 = ctx.Output<Tensor>("Mean1Out");
      Tensor* running_var1 = ctx.Output<Tensor>("Var1Out");

      auto saved_mean1_data = saved_mean1->mutable_data<float>(place);
      auto saved_invstd1_data = saved_invstd1->mutable_data<float>(place);
      auto running_mean1_data = running_mean1->mutable_data<float>(place);
      auto running_var1_data = running_var1->mutable_data<float>(place);

      r = xpu::batch_norm_fusion<XPUT>(dev_ctx.x_context(),
                                       conv1_output_data,
                                       conv2_input_data,
                                       attr.conv1_output_shape[0],
                                       attr.conv1_output_shape[1],
                                       attr.conv1_output_shape[2],
                                       attr.conv1_output_shape[3],
                                       attr.eps,
                                       attr.momentum,
                                       scale1_data,
                                       bias1_data,
                                       saved_mean1_data,
                                       saved_invstd1_data,
                                       running_mean1_data,
                                       running_var1_data,
                                       true,
                                       nullptr,
                                       xpu::Activation_t::RELU,
                                       nullptr,
                                       0);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "batch_norm_fusion");
    } else {
      // bn --> relu
      auto bn1_output_data = RAII_GUARD.alloc<XPUT>(attr.conv1_output_numel);
      PADDLE_ENFORCE_XDNN_NOT_NULL(bn1_output_data);

      const auto* mean1 = ctx.Input<Tensor>("Mean1");
      const auto* var1 = ctx.Input<Tensor>("Var1");
      const auto* mean_data = mean1->data<float>();
      const auto* variance_data = var1->data<float>();
      r = xpu::batch_norm_infer<XPUT>(dev_ctx.x_context(),
                                      conv1_output_data,
                                      bn1_output_data,
                                      attr.conv1_output_shape[0],
                                      attr.conv1_output_shape[1],
                                      attr.conv1_output_shape[2],
                                      attr.conv1_output_shape[3],
                                      attr.eps,
                                      scale1_data,
                                      bias1_data,
                                      mean_data,
                                      variance_data,
                                      true);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "batch_norm_infer");

      r = xpu::relu(dev_ctx.x_context(),
                    bn1_output_data,
                    conv2_input_data,
                    attr.conv1_output_numel);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "relu");
    }

    // 4. conv2
    XPUT* conv2_input_l3_data = nullptr;
    XPUT* conv2_filter_l3_data =
        RAII_GUARD.alloc_l3<XPUT>(attr.conv2_filter_numel);
    if (attr.find_max) {
      Tensor* max_input2 = ctx.Output<Tensor>("MaxInput2");
      Tensor* max_filter2 = ctx.Output<Tensor>("MaxFilter2");
      conv2_input_max_data = max_input2->mutable_data<float>(place);
      conv2_filter_max_data = max_filter2->mutable_data<float>(place);

      r = xpu::findmax_copy_fusion(dev_ctx.x_context(),
                                   conv2_input_data,
                                   conv2_input_max_data,
                                   conv2_input_l3_data,
                                   attr.conv2_input_numel);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "findmax_copy_fusion");

      r = xpu::findmax_copy_fusion(dev_ctx.x_context(),
                                   conv2_filter_data,
                                   conv2_filter_max_data,
                                   conv2_filter_l3_data,
                                   attr.conv2_filter_numel);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "findmax_copy_fusion");
    }
    xpu_conv2d(
        dev_ctx.x_context(),
        conv2_input_l3_data != nullptr ? conv2_input_l3_data : conv2_input_data,
        conv2_filter_l3_data,
        conv2_output_data,
        conv2_input_max_data,
        conv2_filter_max_data,
        attr.conv2_input_shape,
        attr.conv2_filter_shape,
        attr.padding2,
        attr.stride2,
        attr.dilation2,
        attr.group);

    // 5. bn2
    if (!attr.global_stats) {
      Tensor* saved_mean2 = ctx.Output<Tensor>("SavedMean2");
      Tensor* saved_var2 = ctx.Output<Tensor>("SavedInvstd2");
      Tensor* running_mean2 = ctx.Output<Tensor>("Mean2Out");
      Tensor* running_var2 = ctx.Output<Tensor>("Var2Out");

      auto saved_mean2_data = saved_mean2->mutable_data<float>(place);
      auto saved_var2_data = saved_var2->mutable_data<float>(place);
      auto running_mean2_data = running_mean2->mutable_data<float>(place);
      auto running_var2_data = running_var2->mutable_data<float>(place);

      r = xpu::batch_norm_fusion<XPUT>(dev_ctx.x_context(),
                                       conv2_output_data,
                                       output_data,
                                       attr.conv2_output_shape[0],
                                       attr.conv2_output_shape[1],
                                       attr.conv2_output_shape[2],
                                       attr.conv2_output_shape[3],
                                       attr.eps,
                                       attr.momentum,
                                       scale2_data,
                                       bias2_data,
                                       saved_mean2_data,
                                       saved_var2_data,
                                       running_mean2_data,
                                       running_var2_data,
                                       true,
                                       z_out_data,
                                       xpu::Activation_t::RELU,
                                       nullptr,
                                       0);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "batch_norm_fusion");
    } else {
      auto bn2_out_data = RAII_GUARD.alloc<XPUT>(attr.conv2_output_numel);
      PADDLE_ENFORCE_XDNN_NOT_NULL(bn2_out_data);

      const auto* mean2 = ctx.Input<Tensor>("Mean2");
      const auto* var2 = ctx.Input<Tensor>("Var2");
      const auto* mean_data = mean2->data<float>();
      const auto* variance_data = var2->data<float>();
      r = xpu::batch_norm_infer<XPUT>(dev_ctx.x_context(),
                                      conv2_output_data,
                                      bn2_out_data,
                                      attr.conv2_output_shape[0],
                                      attr.conv2_output_shape[1],
                                      attr.conv2_output_shape[2],
                                      attr.conv2_output_shape[3],
                                      attr.eps,
                                      scale2_data,
                                      bias2_data,
                                      mean_data,
                                      variance_data,
                                      true);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "batch_norm_infer");

      r = xpu::add_activation_fusion<XPUT>(dev_ctx.x_context(),
                                           bn2_out_data,
                                           z_out_data,
                                           output_data,
                                           output->numel(),
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           xpu::Activation_t::RELU);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "add_activation_fusion");
    }
  }
};

template <typename T>
class ResNetBasicBlockGradXPUKernel : public framework::OpKernel<T> {
 public:
  using XPUT = typename XPUTypeTrait<T>::Type;

  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_xpu_place(ctx.GetPlace()),
        true,
        platform::errors::PreconditionNotMet("It must use XPUPlace."));

    const Tensor* y_grad = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const Tensor* y = ctx.Input<Tensor>("Y");

    const Tensor* x = ctx.Input<Tensor>("X");
    const Tensor* filter1 = ctx.Input<Tensor>("Filter1");
    const Tensor* scale1 = ctx.Input<Tensor>("Scale1");
    const Tensor* filter2 = ctx.Input<Tensor>("Filter2");
    const Tensor* scale2 = ctx.Input<Tensor>("Scale2");
    const Tensor* saved_mean1 = ctx.Input<Tensor>("SavedMean1");
    const Tensor* saved_invstd1 = ctx.Input<Tensor>("SavedInvstd1");
    const Tensor* saved_mean2 = ctx.Input<Tensor>("SavedMean2");
    const Tensor* saved_invstd2 = ctx.Input<Tensor>("SavedInvstd2");
    const Tensor* conv1_out = ctx.Input<Tensor>("Conv1");
    const Tensor* conv2_out = ctx.Input<Tensor>("Conv2");
    const Tensor* conv2_input = ctx.Input<Tensor>("Conv2Input");

    const Tensor* filter3 = ctx.Input<Tensor>("Filter3");
    const Tensor* conv3_out = ctx.Input<Tensor>("Conv3");
    const Tensor* scale3 = ctx.Input<Tensor>("Scale3");
    const Tensor* saved_mean3 = ctx.Input<Tensor>("SavedMean3");
    const Tensor* saved_invstd3 = ctx.Input<Tensor>("SavedInvstd3");

    const Tensor* conv1_input_max = ctx.Input<Tensor>("MaxInput1");
    const Tensor* conv1_filter_max = ctx.Input<Tensor>("MaxFilter1");
    const Tensor* conv2_input_max = ctx.Input<Tensor>("MaxInput2");
    const Tensor* conv2_filter_max = ctx.Input<Tensor>("MaxFilter2");
    const Tensor* conv3_input_max = ctx.Input<Tensor>("MaxInput3");
    const Tensor* conv3_filter_max = ctx.Input<Tensor>("MaxFilter3");

    Tensor* x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    Tensor* filter1_grad =
        ctx.Output<Tensor>(framework::GradVarName("Filter1"));
    Tensor* scale1_grad = ctx.Output<Tensor>(framework::GradVarName("Scale1"));
    Tensor* bias1_grad = ctx.Output<Tensor>(framework::GradVarName("Bias1"));
    Tensor* filter2_grad =
        ctx.Output<Tensor>(framework::GradVarName("Filter2"));
    Tensor* scale2_grad = ctx.Output<Tensor>(framework::GradVarName("Scale2"));
    Tensor* bias2_grad = ctx.Output<Tensor>(framework::GradVarName("Bias2"));
    Tensor* filter3_grad =
        ctx.Output<Tensor>(framework::GradVarName("Filter3"));
    Tensor* scale3_grad = ctx.Output<Tensor>(framework::GradVarName("Scale3"));
    Tensor* bias3_grad = ctx.Output<Tensor>(framework::GradVarName("Bias3"));

    // attrs
    ResnetBasicBlockGradAttr attr(ctx);
    auto place = ctx.GetPlace();

    const auto* y_grad_data = reinterpret_cast<const XPUT*>(y_grad->data<T>());
    const auto* y_data = reinterpret_cast<const XPUT*>(y->data<T>());
    const auto* x_data = reinterpret_cast<const XPUT*>(x->data<T>());
    const auto* conv1_output_data =
        reinterpret_cast<const XPUT*>(conv1_out->data<T>());
    const auto* conv1_filter_data =
        reinterpret_cast<const XPUT*>(filter1->data<T>());
    const auto* conv2_input_data =
        reinterpret_cast<const XPUT*>(conv2_input->data<T>());
    const auto* conv2_output_data =
        reinterpret_cast<const XPUT*>(conv2_out->data<T>());
    const auto* conv2_filter_data =
        reinterpret_cast<const XPUT*>(filter2->data<T>());

    const auto* scale2_data = scale2->data<float>();
    const auto* saved_mean2_data = saved_mean2->data<float>();
    const auto* saved_invstd2_data = saved_invstd2->data<float>();
    const auto* scale1_data = scale1->data<float>();
    const auto* saved_mean1_data = saved_mean1->data<float>();
    const auto* saved_invstd1_data = saved_invstd1->data<float>();
    auto* scale2_grad_data = scale2_grad->mutable_data<float>(place);
    auto* bias2_grad_data = bias2_grad->mutable_data<float>(place);

    const float* conv1_input_max_data = nullptr;
    const float* conv1_filter_max_data = nullptr;
    const float* conv2_input_max_data = nullptr;
    const float* conv2_filter_max_data = nullptr;
    const float* conv3_input_max_data = nullptr;
    const float* conv3_filter_max_data = nullptr;
    if (attr.find_max) {
      conv1_input_max_data =
          reinterpret_cast<const float*>(conv1_input_max->data<float>());
      conv1_filter_max_data =
          reinterpret_cast<const float*>(conv1_filter_max->data<float>());
      conv2_input_max_data =
          reinterpret_cast<const float*>(conv2_input_max->data<float>());
      conv2_filter_max_data =
          reinterpret_cast<const float*>(conv2_filter_max->data<float>());
      if (attr.has_shortcut) {
        conv3_input_max_data =
            reinterpret_cast<const float*>(conv3_input_max->data<float>());
        conv3_filter_max_data =
            reinterpret_cast<const float*>(conv3_filter_max->data<float>());
      }
    }

    auto& dev_ctx = ctx.template device_context<platform::XPUDeviceContext>();
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    int r = XPU_SUCCESS;

    // 0. bn2, bn2_fusion grad
    auto conv2_output_grad_data =
        RAII_GUARD.alloc<XPUT>(attr.conv2_output_numel);
    PADDLE_ENFORCE_XDNN_NOT_NULL(conv2_output_grad_data);

    XPUT* z_output_grad_data = nullptr;
    XPUT* z_grad_data = nullptr;
    if (!attr.has_shortcut) {
      z_output_grad_data = RAII_GUARD.alloc<XPUT>(attr.conv1_input_numel);
      PADDLE_ENFORCE_XDNN_NOT_NULL(z_output_grad_data);
      z_grad_data = z_output_grad_data;
    } else {
      z_output_grad_data = RAII_GUARD.alloc<XPUT>(attr.conv3_output_numel);
      PADDLE_ENFORCE_XDNN_NOT_NULL(z_output_grad_data);

      z_grad_data = RAII_GUARD.alloc<XPUT>(attr.conv1_input_numel);
      PADDLE_ENFORCE_XDNN_NOT_NULL(z_grad_data);
    }

    r = xpu::batch_norm_grad_fusion<XPUT>(dev_ctx.x_context(),
                                          conv2_output_data,
                                          y_data,
                                          y_grad_data,
                                          conv2_output_grad_data,
                                          attr.conv2_output_shape[0],
                                          attr.conv2_output_shape[1],
                                          attr.conv2_output_shape[2],
                                          attr.conv2_output_shape[3],
                                          scale2_data,
                                          saved_mean2_data,
                                          saved_invstd2_data,
                                          scale2_grad_data,
                                          bias2_grad_data,
                                          true,
                                          z_output_grad_data,
                                          xpu::Activation_t::RELU,
                                          nullptr,
                                          0);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "batch_norm_grad_fusion");

    if (attr.has_shortcut) {
      // bn3 grad
      const auto* conv3_output_data =
          reinterpret_cast<const XPUT*>(conv3_out->data<T>());
      const auto* scale3_data = scale3->data<float>();
      const auto* saved_mean3_data = saved_mean3->data<float>();
      const auto* saved_invstd3_data = saved_invstd3->data<float>();
      auto* scale3_grad_data = scale3_grad->mutable_data<float>(place);
      auto* bias3_grad_data = bias3_grad->mutable_data<float>(place);
      auto* conv3_output_grad_data =
          RAII_GUARD.alloc<XPUT>(attr.conv3_output_numel);

      r = xpu::batch_norm_grad<XPUT>(dev_ctx.x_context(),
                                     conv3_output_data,
                                     z_output_grad_data,
                                     conv3_output_grad_data,
                                     attr.conv3_output_shape[0],
                                     attr.conv3_output_shape[1],
                                     attr.conv3_output_shape[2],
                                     attr.conv3_output_shape[3],
                                     scale3_data,
                                     saved_mean3_data,
                                     saved_invstd3_data,
                                     scale3_grad_data,
                                     bias3_grad_data,
                                     true);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "batch_norm_grad");

      // conv3 grad
      auto* conv3_filter_grad_data =
          reinterpret_cast<XPUT*>(filter3_grad->mutable_data<T>(place));
      auto* conv3_filter_data =
          reinterpret_cast<const XPUT*>(filter3->data<T>());
      xpu_conv2d_grad(dev_ctx.x_context(),
                      x_data,
                      conv3_filter_data,
                      conv3_output_grad_data,
                      z_grad_data,
                      conv3_filter_grad_data,
                      conv3_input_max_data,
                      conv3_filter_max_data,
                      attr.conv3_input_shape,
                      attr.conv3_filter_shape,
                      attr.padding3,
                      attr.stride3,
                      attr.dilation3,
                      attr.group);
    }

    // 2. conv2_grad
    auto* conv2_filter_grad_data =
        reinterpret_cast<XPUT*>(filter2_grad->mutable_data<T>(place));
    auto* conv2_input_grad_data =
        RAII_GUARD.alloc<XPUT>(attr.conv2_input_numel);
    xpu_conv2d_grad(dev_ctx.x_context(),
                    conv2_input_data,
                    conv2_filter_data,
                    conv2_output_grad_data,
                    conv2_input_grad_data,
                    conv2_filter_grad_data,
                    conv2_input_max_data,
                    conv2_filter_max_data,
                    attr.conv2_input_shape,
                    attr.conv2_filter_shape,
                    attr.padding2,
                    attr.stride2,
                    attr.dilation2,
                    attr.group);

    // 3. b1 grad
    auto* conv1_output_grad_data =
        RAII_GUARD.alloc<XPUT>(attr.conv1_output_numel);
    PADDLE_ENFORCE_XDNN_NOT_NULL(conv1_output_grad_data);
    auto* scale1_grad_data = scale1_grad->mutable_data<float>(ctx.GetPlace());
    auto* bias1_grad_data = bias1_grad->mutable_data<float>(ctx.GetPlace());
    r = xpu::batch_norm_grad_fusion<XPUT>(dev_ctx.x_context(),
                                          conv1_output_data,
                                          conv2_input_data,
                                          conv2_input_grad_data,
                                          conv1_output_grad_data,
                                          attr.conv1_output_shape[0],
                                          attr.conv1_output_shape[1],
                                          attr.conv1_output_shape[2],
                                          attr.conv1_output_shape[3],
                                          scale1_data,
                                          saved_mean1_data,
                                          saved_invstd1_data,
                                          scale1_grad_data,
                                          bias1_grad_data,
                                          true,
                                          nullptr,
                                          xpu::Activation_t::RELU,
                                          nullptr,
                                          0);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "batch_norm_grad_fusion");

    // 4. conv1_grad
    auto* x_grad_data = reinterpret_cast<XPUT*>(x_grad->mutable_data<T>(place));
    auto* conv1_filter_grad_data =
        reinterpret_cast<XPUT*>(filter1_grad->mutable_data<T>(place));
    xpu_conv2d_grad(dev_ctx.x_context(),
                    x_data,
                    conv1_filter_data,
                    conv1_output_grad_data,
                    x_grad_data,
                    conv1_filter_grad_data,
                    conv1_input_max_data,
                    conv1_filter_max_data,
                    attr.conv1_input_shape,
                    attr.conv1_filter_shape,
                    attr.padding1,
                    attr.stride1,
                    attr.dilation1,
                    attr.group);

    // add z_grad to x_grad
    r = xpu::add<XPUT>(
        dev_ctx.x_context(), x_grad_data, z_grad_data, x_grad_data, x->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_XPU_KERNEL(resnet_basic_block,
                       ops::ResNetBasicBlockXPUKernel<float>);
REGISTER_OP_XPU_KERNEL(resnet_basic_block_grad,
                       ops::ResNetBasicBlockGradXPUKernel<float>);
#endif
