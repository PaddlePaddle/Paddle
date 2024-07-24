// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_header.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace fusion {
class ResnetBasicBlockAttr {
 public:
  explicit ResnetBasicBlockAttr(const XPUContext &dev_ctx,
                                const DenseTensor &x_in,
                                const DenseTensor &filter1_in,
                                const DenseTensor &scale1_in,
                                const DenseTensor &bias1_in,
                                const DenseTensor &mean1_in,
                                const DenseTensor &var1_in,
                                const DenseTensor &filter2_in,
                                const DenseTensor &scale2_in,
                                const DenseTensor &bias2_in,
                                const DenseTensor &mean2_in,
                                const DenseTensor &var2_in,
                                const paddle::optional<DenseTensor> &filter3_in,
                                const paddle::optional<DenseTensor> &scale3_in,
                                const paddle::optional<DenseTensor> &bias3_in,
                                const paddle::optional<DenseTensor> &mean3_in,
                                const paddle::optional<DenseTensor> &var3_in,
                                int stride1_in,
                                int stride2_in,
                                int stride3_in,
                                int padding1_in,
                                int padding2_in,
                                int padding3_in,
                                int dilation1_in,
                                int dilation2_in,
                                int dilation3_in,
                                int group_in,
                                float momentum_in,
                                float epsilon_in,
                                const std::string &data_format_in,
                                bool has_shortcut_in,
                                bool use_global_stats_in,
                                bool is_test_in,
                                bool trainable_statistics_in,
                                const std::string &act_type_in,
                                bool find_conv_input_max_in,
                                DenseTensor *out,
                                DenseTensor *conv1,
                                DenseTensor *saved_mean1,
                                DenseTensor *saved_invstd1,
                                DenseTensor *mean1_out,
                                DenseTensor *var1_out,
                                DenseTensor *conv2,
                                DenseTensor *conv2_input,
                                DenseTensor *saved_mean2,
                                DenseTensor *saved_invstd2,
                                DenseTensor *mean2_out,
                                DenseTensor *var2_out,
                                DenseTensor *conv3,
                                DenseTensor *saved_mean3,
                                DenseTensor *saved_invstd3,
                                DenseTensor *mean3_out,
                                DenseTensor *var3_out,
                                DenseTensor *max_input1,
                                DenseTensor *max_filter1,
                                DenseTensor *max_input2,
                                DenseTensor *max_filter2,
                                DenseTensor *max_input3,
                                DenseTensor *max_filter3) {
    padding1 = padding1_in;
    padding2 = padding2_in;
    padding3 = padding3_in;
    stride1 = stride1_in;
    stride2 = stride2_in;
    stride3 = stride3_in;
    dilation1 = dilation1_in;
    dilation2 = dilation2_in;
    dilation3 = dilation3_in;
    group = group_in;

    eps = static_cast<double>(epsilon_in);
    momentum = static_cast<double>(momentum_in);
    has_shortcut = has_shortcut_in;
    find_max = find_conv_input_max_in;

    const auto is_test = is_test_in;
    const auto use_global_stats = use_global_stats_in;
    const auto trainable_stats = trainable_statistics_in;

    bool test_mode = is_test && (!trainable_stats);
    global_stats = test_mode || use_global_stats;

    // init shape
    auto input1 = &x_in;
    auto filter1 = &filter1_in;
    auto conv1_out = conv1;
    auto filter2 = &filter2_in;
    auto conv2_out = conv2;
    conv1_input_shape = common::vectorize<int>(input1->dims());
    conv1_output_shape = common::vectorize<int>(conv1_out->dims());
    conv1_filter_shape = common::vectorize<int>(filter1->dims());
    conv1_filter_numel = filter1->numel();
    conv1_input_numel = input1->numel();
    conv1_output_numel = conv1_out->numel();

    conv2_input_shape = common::vectorize<int>(conv1_out->dims());
    conv2_output_shape = common::vectorize<int>(conv2_out->dims());
    conv2_filter_shape = common::vectorize<int>(filter2->dims());
    conv2_filter_numel = filter2->numel();
    conv2_input_numel = conv1_out->numel();
    conv2_output_numel = conv2_out->numel();

    if (has_shortcut) {
      auto filter3 = filter3_in.get_ptr();
      auto conv3_out = conv3;
      conv3_input_shape = common::vectorize<int>(input1->dims());
      conv3_output_shape = common::vectorize<int>(conv3_out->dims());
      conv3_filter_shape = common::vectorize<int>(filter3->dims());
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

template <typename T>
static inline void xpu_conv2d(xpu::Context *ctx,
                              const T *input_data,
                              const T *filter_data,
                              T *output_data,
                              float *input_max_data,
                              float *filter_max_data,
                              const std::vector<int> &input_shape,
                              const std::vector<int> &filter_shape,
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

template <typename T, typename Context>
void ResNetBasicBlockXPUKernel(const Context &dev_ctx,
                               const DenseTensor &x_in,
                               const DenseTensor &filter1_in,
                               const DenseTensor &scale1_in,
                               const DenseTensor &bias1_in,
                               const DenseTensor &mean1_in,
                               const DenseTensor &var1_in,
                               const DenseTensor &filter2_in,
                               const DenseTensor &scale2_in,
                               const DenseTensor &bias2_in,
                               const DenseTensor &mean2_in,
                               const DenseTensor &var2_in,
                               const paddle::optional<DenseTensor> &filter3_in,
                               const paddle::optional<DenseTensor> &scale3_in,
                               const paddle::optional<DenseTensor> &bias3_in,
                               const paddle::optional<DenseTensor> &mean3_in,
                               const paddle::optional<DenseTensor> &var3_in,
                               int stride1,
                               int stride2,
                               int stride3,
                               int padding1,
                               int padding2,
                               int padding3,
                               int dilation1,
                               int dilation2,
                               int dilation3,
                               int group,
                               float momentum_in,
                               float epsilon,
                               const std::string &data_format,
                               bool has_shortcut,
                               bool use_global_stats,
                               bool is_test,
                               bool trainable_statistics,
                               const std::string &act_type,
                               bool find_conv_input_max,
                               DenseTensor *out,
                               DenseTensor *conv1,
                               DenseTensor *saved_mean1,
                               DenseTensor *saved_invstd1,
                               DenseTensor *mean1_out,
                               DenseTensor *var1_out,
                               DenseTensor *conv2,
                               DenseTensor *conv2_input,
                               DenseTensor *saved_mean2,
                               DenseTensor *saved_invstd2,
                               DenseTensor *mean2_out,
                               DenseTensor *var2_out,
                               DenseTensor *conv3,
                               DenseTensor *saved_mean3,
                               DenseTensor *saved_invstd3,
                               DenseTensor *mean3_out,
                               DenseTensor *var3_out,
                               DenseTensor *max_input1,
                               DenseTensor *max_filter1,
                               DenseTensor *max_input2,
                               DenseTensor *max_filter2,
                               DenseTensor *max_input3,
                               DenseTensor *max_filter3) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  // input
  const phi::DenseTensor *x = &x_in;
  const phi::DenseTensor *filter1 = &filter1_in;
  const phi::DenseTensor *scale1 = &scale1_in;
  const phi::DenseTensor *bias1 = &bias1_in;
  const phi::DenseTensor *filter2 = &filter2_in;
  const phi::DenseTensor *scale2 = &scale2_in;
  const phi::DenseTensor *bias2 = &bias2_in;

  // output
  phi::DenseTensor *conv1_output = conv1;
  phi::DenseTensor *conv2_output = conv2;
  phi::DenseTensor *output = out;

  auto x_data = reinterpret_cast<const XPUType *>(x->data<T>());
  auto conv1_filter_data =
      reinterpret_cast<const XPUType *>(filter1->data<T>());
  auto conv2_filter_data =
      reinterpret_cast<const XPUType *>(filter2->data<T>());
  auto conv1_output_data =
      reinterpret_cast<XPUType *>(dev_ctx.template Alloc<T>(conv1_output));
  auto conv2_input_data =
      reinterpret_cast<XPUType *>(dev_ctx.template Alloc<T>(conv2_input));
  auto conv2_output_data =
      reinterpret_cast<XPUType *>(dev_ctx.template Alloc<T>(conv2_output));
  auto scale1_data = scale1->data<float>();
  auto scale2_data = scale2->data<float>();
  auto bias1_data = bias1->data<float>();
  auto bias2_data = bias2->data<float>();
  auto output_data =
      reinterpret_cast<XPUType *>(dev_ctx.template Alloc<T>(output));

  float *conv1_input_max_data = nullptr;
  float *conv1_filter_max_data = nullptr;
  float *conv2_input_max_data = nullptr;
  float *conv2_filter_max_data = nullptr;
  float *conv3_input_max_data = nullptr;
  float *conv3_filter_max_data = nullptr;

  ResnetBasicBlockAttr attr(dev_ctx,
                            x_in,
                            filter1_in,
                            scale1_in,
                            bias1_in,
                            mean1_in,
                            var1_in,
                            filter2_in,
                            scale2_in,
                            bias2_in,
                            mean2_in,
                            var2_in,
                            filter3_in,
                            scale3_in,
                            bias3_in,
                            mean3_in,
                            var3_in,
                            stride1,
                            stride2,
                            stride3,
                            padding1,
                            padding2,
                            padding3,
                            dilation1,
                            dilation2,
                            dilation3,
                            group,
                            momentum_in,
                            epsilon,
                            data_format,
                            has_shortcut,
                            use_global_stats,
                            is_test,
                            trainable_statistics,
                            act_type,
                            find_conv_input_max,
                            out,
                            conv1,
                            saved_mean1,
                            saved_invstd1,
                            mean1_out,
                            var1_out,
                            conv2,
                            conv2_input,
                            saved_mean2,
                            saved_invstd2,
                            mean2_out,
                            var2_out,
                            conv3,
                            saved_mean3,
                            saved_invstd3,
                            mean3_out,
                            var3_out,
                            max_input1,
                            max_filter1,
                            max_input2,
                            max_filter2,
                            max_input3,
                            max_filter3);

  // init find max
  if (attr.find_max) {
    conv1_input_max_data = dev_ctx.template Alloc<float>(max_input1);
    conv1_filter_max_data = dev_ctx.template Alloc<float>(max_filter1);

    conv2_input_max_data = dev_ctx.template Alloc<float>(max_input2);
    conv2_filter_max_data = dev_ctx.template Alloc<float>(max_filter2);

    if (attr.has_shortcut) {
      conv3_input_max_data = dev_ctx.template Alloc<float>(max_input3);
      conv3_filter_max_data = dev_ctx.template Alloc<float>(max_filter3);
    }
  }

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  int r = XPU_SUCCESS;

  // 1. short
  const XPUType *z_out_data = nullptr;
  if (attr.has_shortcut) {
    phi::DenseTensor *conv3_out = conv3;
    const phi::DenseTensor *filter3 = filter3_in.get_ptr();
    auto conv3_filter_data =
        reinterpret_cast<const XPUType *>(filter3->data<T>());
    auto conv3_output_data =
        reinterpret_cast<XPUType *>(dev_ctx.template Alloc<T>(conv3_out));

    XPUType *conv3_input_l3_data = nullptr;
    XPUType *conv3_filter_l3_data =
        RAII_GUARD.alloc_l3_or_gm<XPUType>(attr.conv3_filter_numel);

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
    const phi::DenseTensor *scale3 = scale3_in.get_ptr();
    const phi::DenseTensor *bias3 = bias3_in.get_ptr();
    auto bias3_data = bias3->data<float>();
    auto scale3_data = scale3->data<float>();

    auto bn3_output_data = RAII_GUARD.alloc<XPUType>(attr.conv3_output_numel);
    PADDLE_ENFORCE_XDNN_NOT_NULL(bn3_output_data);

    if (!attr.global_stats) {
      phi::DenseTensor *running_mean3 = mean3_out;
      phi::DenseTensor *running_var3 = var3_out;

      auto saved_mean3_data = dev_ctx.template Alloc<float>(saved_mean3);
      auto saved_invstd3_data = dev_ctx.template Alloc<float>(saved_invstd3);
      auto running_mean3_data = dev_ctx.template Alloc<float>(running_mean3);
      auto running_var3_data = dev_ctx.template Alloc<float>(running_var3);

      r = xpu::batch_norm_fusion<XPUType>(dev_ctx.x_context(),
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
      const auto *mean3 = mean3_in.get_ptr();
      const auto *var3 = var3_in.get_ptr();
      const auto *mean3_data = mean3->data<float>();
      const auto *variance3_data = var3->data<float>();
      r = xpu::batch_norm_infer<XPUType>(dev_ctx.x_context(),
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
    z_out_data = reinterpret_cast<const XPUType *>(bn3_output_data);
  } else {
    z_out_data = x_data;
  }

  // 2. conv1
  XPUType *conv1_input_l3_data = nullptr;
  XPUType *conv1_filter_l3_data =
      RAII_GUARD.alloc_l3_or_gm<XPUType>(attr.conv1_filter_numel);
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
    phi::DenseTensor *running_mean1 = mean1_out;
    phi::DenseTensor *running_var1 = var1_out;

    auto saved_mean1_data = dev_ctx.template Alloc<float>(saved_mean1);
    auto saved_invstd1_data = dev_ctx.template Alloc<float>(saved_invstd1);
    auto running_mean1_data = dev_ctx.template Alloc<float>(running_mean1);
    auto running_var1_data = dev_ctx.template Alloc<float>(running_var1);

    r = xpu::batch_norm_fusion<XPUType>(dev_ctx.x_context(),
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
    auto bn1_output_data = RAII_GUARD.alloc<XPUType>(attr.conv1_output_numel);
    PADDLE_ENFORCE_XDNN_NOT_NULL(bn1_output_data);

    const auto *mean1 = &mean1_in;
    const auto *var1 = &var1_in;
    const auto *mean_data = mean1->data<float>();
    const auto *variance_data = var1->data<float>();
    r = xpu::batch_norm_infer<XPUType>(dev_ctx.x_context(),
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
  XPUType *conv2_input_l3_data = nullptr;
  XPUType *conv2_filter_l3_data =
      RAII_GUARD.alloc_l3_or_gm<XPUType>(attr.conv2_filter_numel);
  if (attr.find_max) {
    conv2_input_max_data = dev_ctx.template Alloc<float>(max_input2);
    conv2_filter_max_data = dev_ctx.template Alloc<float>(max_filter2);

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
    phi::DenseTensor *saved_var2 = saved_invstd2;
    phi::DenseTensor *running_mean2 = mean2_out;
    phi::DenseTensor *running_var2 = var2_out;

    auto saved_mean2_data = dev_ctx.template Alloc<float>(saved_mean2);
    auto saved_var2_data = dev_ctx.template Alloc<float>(saved_var2);
    auto running_mean2_data = dev_ctx.template Alloc<float>(running_mean2);
    auto running_var2_data = dev_ctx.template Alloc<float>(running_var2);

    r = xpu::batch_norm_fusion<XPUType>(dev_ctx.x_context(),
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
    auto bn2_out_data = RAII_GUARD.alloc<XPUType>(attr.conv2_output_numel);
    PADDLE_ENFORCE_XDNN_NOT_NULL(bn2_out_data);

    const auto *mean2 = &mean2_in;
    const auto *var2 = &var2_in;
    const auto *mean_data = mean2->data<float>();
    const auto *variance_data = var2->data<float>();
    r = xpu::batch_norm_infer<XPUType>(dev_ctx.x_context(),
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

    r = xpu::add_activation_fusion<XPUType>(dev_ctx.x_context(),
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
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(resnet_basic_block,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::ResNetBasicBlockXPUKernel,
                   float) {}
