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
void ResNetUnitXPUKernel(const Context &dev_ctx,
                         const DenseTensor &x_in,
                         const DenseTensor &filter_x_in,
                         const DenseTensor &scale_x_in,
                         const DenseTensor &bias_x_in,
                         const DenseTensor &mean_x_in,
                         const DenseTensor &var_x_in,
                         const paddle::optional<DenseTensor> &z_in,
                         const paddle::optional<DenseTensor> &filter_z_in,
                         const paddle::optional<DenseTensor> &scale_z_in,
                         const paddle::optional<DenseTensor> &bias_z_in,
                         const paddle::optional<DenseTensor> &mean_z_in,
                         const paddle::optional<DenseTensor> &var_z_in,
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
                         DenseTensor *out,
                         DenseTensor *bit_mask,
                         DenseTensor *conv_x,
                         DenseTensor *saved_mean_x,
                         DenseTensor *saved_invstd_x,
                         DenseTensor *running_mean_x,
                         DenseTensor *running_var_x,
                         DenseTensor *conv_z,
                         DenseTensor *saved_mean_z,
                         DenseTensor *saved_invstd_z,
                         DenseTensor *running_mean_z,
                         DenseTensor *running_var_z) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  bool is_nchw = (data_format == "NCHW");
  // input x
  const phi::DenseTensor *input_x = &x_in;
  const phi::DenseTensor *filter_x = &filter_x_in;
  const phi::DenseTensor *scale_x = &scale_x_in;
  const phi::DenseTensor *bias_x = &bias_x_in;

  // output x
  phi::DenseTensor *conv_out_x = conv_x;

  phi::DenseTensor *output = out;

  //  attrs
  float eps = epsilon;
  float momentum = momentum_in;
  bool is_train = !is_test && !use_global_stats;

  std::vector<const XPUType *> x_list = {
      reinterpret_cast<const XPUType *>(input_x->data<T>())};
  std::vector<const XPUType *> w_list = {
      reinterpret_cast<const XPUType *>(filter_x->data<T>())};
  std::vector<XPUType *> conv_y_list = {
      reinterpret_cast<XPUType *>(dev_ctx.template Alloc<T>(conv_out_x))};

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
  std::vector<float *> batch_mean_list = {
      dev_ctx.template Alloc<float>(saved_mean_x)};
  std::vector<float *> batch_invstd_list = {
      dev_ctx.template Alloc<float>(saved_invstd_x)};
  std::vector<float *> global_mean_list = {
      dev_ctx.template Alloc<float>(running_mean_x)};
  std::vector<float *> global_var_list = {
      dev_ctx.template Alloc<float>(running_var_x)};

  std::vector<const float *> x_maxlist = {nullptr};
  std::vector<const float *> w_maxlist = {nullptr};
  if (has_shortcut) {
    // input z
    const phi::DenseTensor *input_z = z_in.get_ptr();
    const phi::DenseTensor *filter_z = filter_z_in.get_ptr();
    const phi::DenseTensor *scale_z = scale_z_in.get_ptr();
    const phi::DenseTensor *bias_z = bias_z_in.get_ptr();

    phi::DenseTensor *conv_out_z = conv_z;

    x_list.push_back(reinterpret_cast<const XPUType *>(input_z->data<T>()));
    w_list.push_back(reinterpret_cast<const XPUType *>(filter_z->data<T>()));
    conv_y_list.push_back(
        reinterpret_cast<XPUType *>(dev_ctx.template Alloc<T>(conv_out_z)));

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
    batch_mean_list.push_back(dev_ctx.template Alloc<float>(saved_mean_z));
    batch_invstd_list.push_back(dev_ctx.template Alloc<float>(saved_invstd_z));
    global_mean_list.push_back(dev_ctx.template Alloc<float>(running_mean_z));
    global_var_list.push_back(dev_ctx.template Alloc<float>(running_var_z));
    x_maxlist.push_back(nullptr);
    w_maxlist.push_back(nullptr);
  } else {
    if (fuse_add) {
      const phi::DenseTensor *input_z = z_in.get_ptr();
      auto input_z_shape = common::vectorize<int>(input_z->dims());
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
      reinterpret_cast<XPUType *>(dev_ctx.template Alloc<T>(output)),
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

}  // namespace phi

PD_REGISTER_KERNEL(resnet_unit,
                   XPU,
                   ALL_LAYOUT,
                   phi::ResNetUnitXPUKernel,
                   phi::dtype::float16,
                   float) {}
