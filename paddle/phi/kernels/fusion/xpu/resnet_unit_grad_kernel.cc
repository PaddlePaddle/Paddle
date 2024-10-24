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
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/utils/optional.h"

namespace phi {

template <typename T, typename Context>
void ResNetUnitGradXPUKernel(
    const Context &dev_ctx,
    const DenseTensor &x_in,
    const DenseTensor &filter_x_in,
    const DenseTensor &conv_x_in,
    const DenseTensor &scale_x_in,
    const DenseTensor &bias_x_in,
    const DenseTensor &saved_mean_x_in,
    const DenseTensor &saved_invstd_x_in,
    const paddle::optional<DenseTensor> &z_in,
    const paddle::optional<DenseTensor> &filter_z_in,
    const paddle::optional<DenseTensor> &conv_z_in,
    const paddle::optional<DenseTensor> &scale_z_in,
    const paddle::optional<DenseTensor> &bias_z_in,
    const paddle::optional<DenseTensor> &saved_mean_z_in,
    const paddle::optional<DenseTensor> &saved_invstd_z_in,
    const DenseTensor &out,
    const DenseTensor &bit_mask,
    const DenseTensor &out_grad,
    int stride,
    int stride_z,
    int padding,
    int dilation,
    int group,
    float momentum_in,
    float epsilon,
    const std::string &data_format,
    bool fuse_add,
    bool has_shortcut,
    bool use_global_stats,
    bool is_test,
    bool use_addto,
    const std::string &act_type,
    DenseTensor *x_grad,
    DenseTensor *filter_x_grad,
    DenseTensor *scale_x_grad,
    DenseTensor *bias_x_grad,
    DenseTensor *z_grad,
    DenseTensor *filter_z_grad,
    DenseTensor *scale_z_grad,
    DenseTensor *bias_z_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  bool is_nchw = (data_format == "NCHW");
  const phi::DenseTensor *y_grad = &out_grad;
  const phi::DenseTensor *x = &x_in;
  const phi::DenseTensor *filter_x = &filter_x_in;
  const phi::DenseTensor *scale_x = &scale_x_in;
  const phi::DenseTensor *saved_mean_x = &saved_mean_x_in;
  const phi::DenseTensor *saved_invstd_x = &saved_invstd_x_in;
  const phi::DenseTensor *conv_out_x = &conv_x_in;
  const phi::DenseTensor *output = &out;

  float eps = epsilon;

  std::vector<const XPUType *> x_list = {
      reinterpret_cast<const XPUType *>(x->data<T>())};
  std::vector<const XPUType *> w_list = {
      reinterpret_cast<const XPUType *>(filter_x->data<T>())};
  std::vector<const XPUType *> conv_y_list = {
      reinterpret_cast<const XPUType *>(conv_out_x->data<T>())};
  std::vector<XPUType *> dx_list = {
      reinterpret_cast<XPUType *>(dev_ctx.template Alloc<T>(x_grad))};
  std::vector<XPUType *> dw_list = {
      reinterpret_cast<XPUType *>(dev_ctx.template Alloc<T>(filter_x_grad))};

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

  std::vector<const float *> x_maxlist = {nullptr};
  std::vector<const float *> w_maxlist = {nullptr};

  std::vector<const float *> scale_list = {scale_x->data<float>()};
  std::vector<const float *> batch_mean_list = {saved_mean_x->data<float>()};
  std::vector<const float *> batch_invstd_list = {
      saved_invstd_x->data<float>()};
  std::vector<float *> dscale_list = {
      dev_ctx.template Alloc<float>(scale_x_grad)};
  std::vector<float *> dbias_list = {
      dev_ctx.template Alloc<float>(bias_x_grad)};

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
    const phi::DenseTensor *z = z_in.get_ptr();
    const phi::DenseTensor *filter_z = filter_z_in.get_ptr();
    const phi::DenseTensor *scale_z = scale_z_in.get_ptr();
    const phi::DenseTensor *saved_mean_z = saved_mean_z_in.get_ptr();
    const phi::DenseTensor *saved_invstd_z = saved_invstd_z_in.get_ptr();
    const phi::DenseTensor *conv_out_z = conv_z_in.get_ptr();

    x_list.push_back(reinterpret_cast<const XPUType *>(z->data<T>()));
    w_list.push_back(reinterpret_cast<const XPUType *>(filter_z->data<T>()));
    conv_y_list.push_back(
        reinterpret_cast<const XPUType *>(conv_out_z->data<T>()));
    dx_list.push_back(
        reinterpret_cast<XPUType *>(dev_ctx.template Alloc<T>(z_grad)));
    dw_list.push_back(
        reinterpret_cast<XPUType *>(dev_ctx.template Alloc<T>(filter_z_grad)));
    x_shape_list.push_back(common::vectorize<int>(z->dims()));

    auto filter_z_shape = common::vectorize<int>(filter_z->dims());
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
    dscale_list.push_back(dev_ctx.template Alloc<float>(scale_z_grad));
    dbias_list.push_back(dev_ctx.template Alloc<float>(bias_z_grad));
  } else {
    if (fuse_add) {
      auto z_grad_tmp = z_grad;
      dx_list.push_back(
          reinterpret_cast<XPUType *>(dev_ctx.template Alloc<T>(z_grad_tmp)));
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

}  // namespace phi
PD_REGISTER_KERNEL(resnet_unit_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::ResNetUnitGradXPUKernel,
                   phi::dtype::float16,
                   float) {}
