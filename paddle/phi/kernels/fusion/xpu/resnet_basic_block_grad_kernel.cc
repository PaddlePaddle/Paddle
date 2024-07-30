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

class ResnetBasicBlockGradAttr {
 public:
  explicit ResnetBasicBlockGradAttr(
      const XPUContext &dev_ctx,
      const DenseTensor &x_in,
      const DenseTensor &filter1_in,
      const DenseTensor &conv1_in,
      const DenseTensor &scale1_in,
      const DenseTensor &bias1_in,
      const DenseTensor &saved_mean1_in,
      const DenseTensor &saved_invstd1_in,
      const DenseTensor &filter2_in,
      const DenseTensor &conv2_in,
      const DenseTensor &conv2_input_in,
      const DenseTensor &scale2_in,
      const DenseTensor &bias2_in,
      const DenseTensor &saved_mean2_in,
      const DenseTensor &saved_invstd2_in,
      const paddle::optional<DenseTensor> &filter3_in,
      const paddle::optional<DenseTensor> &conv3_in,
      const paddle::optional<DenseTensor> &scale3_in,
      const paddle::optional<DenseTensor> &bias3_in,
      const paddle::optional<DenseTensor> &saved_mean3_in,
      const paddle::optional<DenseTensor> &saved_invstd3_in,
      const DenseTensor &max_input1,
      const DenseTensor &max_filter1,
      const DenseTensor &max_input2,
      const DenseTensor &max_filter2,
      const DenseTensor &max_input3,
      const DenseTensor &max_filter3,
      const DenseTensor &out,
      const DenseTensor &out_grad,
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
      DenseTensor *x_grad,
      DenseTensor *filter1_grad,
      DenseTensor *scale1_grad,
      DenseTensor *bias1_grad,
      DenseTensor *filter2_grad,
      DenseTensor *scale2_grad,
      DenseTensor *bias2_grad,
      DenseTensor *filter3_grad,
      DenseTensor *scale3_grad,
      DenseTensor *bias3_grad) {
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

    has_shortcut = has_shortcut_in;
    find_max = find_conv_input_max_in;

    // init shape
    auto input1 = &x_in;
    auto filter1 = &filter1_in;
    auto conv1_out = &conv1_in;
    auto filter2 = &filter2_in;
    auto conv2_out = &conv2_in;
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
      auto conv3_out = conv3_in.get_ptr();
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
static inline void xpu_conv2d_grad(xpu::Context *ctx,
                                   const T *input_data,
                                   const T *filter_data,
                                   const T *output_grad_data,
                                   T *input_grad_data,
                                   T *filter_grad_data,
                                   const float *input_max_data,
                                   const float *filter_max_data,
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

template <typename T, typename Context>
void ResNetBasicBlockGradXPUKernel(
    const Context &dev_ctx,
    const DenseTensor &x_in,
    const DenseTensor &filter1_in,
    const DenseTensor &conv1_in,
    const DenseTensor &scale1_in,
    const DenseTensor &bias1_in,
    const DenseTensor &saved_mean1_in,
    const DenseTensor &saved_invstd1_in,
    const DenseTensor &filter2_in,
    const DenseTensor &conv2_in,
    const DenseTensor &conv2_input_in,
    const DenseTensor &scale2_in,
    const DenseTensor &bias2_in,
    const DenseTensor &saved_mean2_in,
    const DenseTensor &saved_invstd2_in,
    const paddle::optional<DenseTensor> &filter3_in,
    const paddle::optional<DenseTensor> &conv3_in,
    const paddle::optional<DenseTensor> &scale3_in,
    const paddle::optional<DenseTensor> &bias3_in,
    const paddle::optional<DenseTensor> &saved_mean3_in,
    const paddle::optional<DenseTensor> &saved_invstd3_in,
    const DenseTensor &max_input1,
    const DenseTensor &max_filter1,
    const DenseTensor &max_input2,
    const DenseTensor &max_filter2,
    const DenseTensor &max_input3,
    const DenseTensor &max_filter3,
    const DenseTensor &out,
    const DenseTensor &out_grad,
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
    DenseTensor *x_grad,
    DenseTensor *filter1_grad,
    DenseTensor *scale1_grad,
    DenseTensor *bias1_grad,
    DenseTensor *filter2_grad,
    DenseTensor *scale2_grad,
    DenseTensor *bias2_grad,
    DenseTensor *filter3_grad,
    DenseTensor *scale3_grad,
    DenseTensor *bias3_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  const phi::DenseTensor *y_grad = &out_grad;
  const phi::DenseTensor *y = &out;

  const phi::DenseTensor *x = &x_in;
  const phi::DenseTensor *filter1 = &filter1_in;
  const phi::DenseTensor *scale1 = &scale1_in;
  const phi::DenseTensor *filter2 = &filter2_in;
  const phi::DenseTensor *scale2 = &scale2_in;
  const phi::DenseTensor *saved_mean1 = &saved_mean1_in;
  const phi::DenseTensor *saved_invstd1 = &saved_invstd1_in;
  const phi::DenseTensor *saved_mean2 = &saved_mean2_in;
  const phi::DenseTensor *saved_invstd2 = &saved_invstd2_in;
  const phi::DenseTensor *conv1_out = &conv1_in;
  const phi::DenseTensor *conv2_out = &conv2_in;
  const phi::DenseTensor *conv2_input = &conv2_input_in;

  const phi::DenseTensor *filter3 = filter3_in.get_ptr();
  const phi::DenseTensor *conv3_out = conv3_in.get_ptr();
  const phi::DenseTensor *scale3 = scale3_in.get_ptr();
  const phi::DenseTensor *saved_mean3 = saved_mean3_in.get_ptr();
  const phi::DenseTensor *saved_invstd3 = saved_invstd3_in.get_ptr();

  const phi::DenseTensor *conv1_input_max = &max_input1;
  const phi::DenseTensor *conv1_filter_max = &max_filter1;
  const phi::DenseTensor *conv2_input_max = &max_input2;
  const phi::DenseTensor *conv2_filter_max = &max_filter2;
  const phi::DenseTensor *conv3_input_max = &max_input3;
  const phi::DenseTensor *conv3_filter_max = &max_filter3;

  // attrs
  ResnetBasicBlockGradAttr attr(dev_ctx,
                                x_in,
                                filter1_in,
                                conv1_in,
                                scale1_in,
                                bias1_in,
                                saved_mean1_in,
                                saved_invstd1_in,
                                filter2_in,
                                conv2_in,
                                conv2_input_in,
                                scale2_in,
                                bias2_in,
                                saved_mean2_in,
                                saved_invstd2_in,
                                filter3_in,
                                conv3_in,
                                scale3_in,
                                bias3_in,
                                saved_mean3_in,
                                saved_invstd3_in,
                                max_input1,
                                max_filter1,
                                max_input2,
                                max_filter2,
                                max_input3,
                                max_filter3,
                                out,
                                out_grad,
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
                                x_grad,
                                filter1_grad,
                                scale1_grad,
                                bias1_grad,
                                filter2_grad,
                                scale2_grad,
                                bias2_grad,
                                filter3_grad,
                                scale3_grad,
                                bias3_grad);
  const auto *y_grad_data =
      reinterpret_cast<const XPUType *>(y_grad->data<T>());
  const auto *y_data = reinterpret_cast<const XPUType *>(y->data<T>());
  const auto *x_data = reinterpret_cast<const XPUType *>(x->data<T>());
  const auto *conv1_output_data =
      reinterpret_cast<const XPUType *>(conv1_out->data<T>());
  const auto *conv1_filter_data =
      reinterpret_cast<const XPUType *>(filter1->data<T>());
  const auto *conv2_input_data =
      reinterpret_cast<const XPUType *>(conv2_input->data<T>());
  const auto *conv2_output_data =
      reinterpret_cast<const XPUType *>(conv2_out->data<T>());
  const auto *conv2_filter_data =
      reinterpret_cast<const XPUType *>(filter2->data<T>());

  const auto *scale2_data = scale2->data<float>();
  const auto *saved_mean2_data = saved_mean2->data<float>();
  const auto *saved_invstd2_data = saved_invstd2->data<float>();
  const auto *scale1_data = scale1->data<float>();
  const auto *saved_mean1_data = saved_mean1->data<float>();
  const auto *saved_invstd1_data = saved_invstd1->data<float>();
  auto *scale2_grad_data = dev_ctx.template Alloc<float>(scale2_grad);
  auto *bias2_grad_data = dev_ctx.template Alloc<float>(bias2_grad);

  const float *conv1_input_max_data = nullptr;
  const float *conv1_filter_max_data = nullptr;
  const float *conv2_input_max_data = nullptr;
  const float *conv2_filter_max_data = nullptr;
  const float *conv3_input_max_data = nullptr;
  const float *conv3_filter_max_data = nullptr;
  if (attr.find_max) {
    conv1_input_max_data =
        reinterpret_cast<const float *>(conv1_input_max->data<float>());
    conv1_filter_max_data =
        reinterpret_cast<const float *>(conv1_filter_max->data<float>());
    conv2_input_max_data =
        reinterpret_cast<const float *>(conv2_input_max->data<float>());
    conv2_filter_max_data =
        reinterpret_cast<const float *>(conv2_filter_max->data<float>());
    if (attr.has_shortcut) {
      conv3_input_max_data =
          reinterpret_cast<const float *>(conv3_input_max->data<float>());
      conv3_filter_max_data =
          reinterpret_cast<const float *>(conv3_filter_max->data<float>());
    }
  }

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  int r = XPU_SUCCESS;

  // 0. bn2, bn2_fusion grad
  auto conv2_output_grad_data =
      RAII_GUARD.alloc<XPUType>(attr.conv2_output_numel);
  PADDLE_ENFORCE_XDNN_NOT_NULL(conv2_output_grad_data);

  XPUType *z_output_grad_data = nullptr;
  XPUType *z_grad_data = nullptr;
  if (!attr.has_shortcut) {
    z_output_grad_data = RAII_GUARD.alloc<XPUType>(attr.conv1_input_numel);
    PADDLE_ENFORCE_XDNN_NOT_NULL(z_output_grad_data);
    z_grad_data = z_output_grad_data;
  } else {
    z_output_grad_data = RAII_GUARD.alloc<XPUType>(attr.conv3_output_numel);
    PADDLE_ENFORCE_XDNN_NOT_NULL(z_output_grad_data);

    z_grad_data = RAII_GUARD.alloc<XPUType>(attr.conv1_input_numel);
    PADDLE_ENFORCE_XDNN_NOT_NULL(z_grad_data);
  }

  r = xpu::batch_norm_grad_fusion<XPUType>(dev_ctx.x_context(),
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
    const auto *conv3_output_data =
        reinterpret_cast<const XPUType *>(conv3_out->data<T>());
    const auto *scale3_data = scale3->data<float>();
    const auto *saved_mean3_data = saved_mean3->data<float>();
    const auto *saved_invstd3_data = saved_invstd3->data<float>();
    auto *scale3_grad_data = dev_ctx.template Alloc<float>(scale3_grad);
    auto *bias3_grad_data = dev_ctx.template Alloc<float>(bias3_grad);
    auto *conv3_output_grad_data =
        RAII_GUARD.alloc<XPUType>(attr.conv3_output_numel);

    r = xpu::batch_norm_grad<XPUType>(dev_ctx.x_context(),
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
    auto *conv3_filter_grad_data =
        reinterpret_cast<XPUType *>(dev_ctx.template Alloc<T>(filter3_grad));
    auto *conv3_filter_data =
        reinterpret_cast<const XPUType *>(filter3->data<T>());
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
  auto *conv2_filter_grad_data =
      reinterpret_cast<XPUType *>(dev_ctx.template Alloc<T>(filter2_grad));
  auto *conv2_input_grad_data =
      RAII_GUARD.alloc<XPUType>(attr.conv2_input_numel);
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
  auto *conv1_output_grad_data =
      RAII_GUARD.alloc<XPUType>(attr.conv1_output_numel);
  PADDLE_ENFORCE_XDNN_NOT_NULL(conv1_output_grad_data);
  auto *scale1_grad_data = dev_ctx.template Alloc<float>(scale1_grad);
  auto *bias1_grad_data = dev_ctx.template Alloc<float>(bias1_grad);
  r = xpu::batch_norm_grad_fusion<XPUType>(dev_ctx.x_context(),
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
  auto *x_grad_data =
      reinterpret_cast<XPUType *>(dev_ctx.template Alloc<T>(x_grad));
  auto *conv1_filter_grad_data =
      reinterpret_cast<XPUType *>(dev_ctx.template Alloc<T>(filter1_grad));
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
  r = xpu::add<XPUType>(
      dev_ctx.x_context(), x_grad_data, z_grad_data, x_grad_data, x->numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");
}
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(resnet_basic_block_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::ResNetBasicBlockGradXPUKernel,
                   float) {}
