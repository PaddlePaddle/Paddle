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

#pragma once

#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/fusion/gpu/cudnn_bn_stats_finalize.cu.h"
#include "paddle/phi/kernels/fusion/gpu/cudnn_norm_conv.cu.h"
#include "paddle/phi/kernels/fusion/gpu/cudnn_scale_bias_add_relu.cu.h"
#include "paddle/utils/optional.h"

#if CUDNN_VERSION >= 8000
namespace phi {

template <typename T, typename Context>
void ResNetUnitKernel(const Context &dev_ctx,
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
  PADDLE_ENFORCE_EQ(phi::backends::gpu::CudnnDataType<T>::type,
                    CUDNN_DATA_HALF,
                    common::errors::Unavailable(
                        "ResNetUnitOp only supports float16 for now."));

  // input x
  const phi::DenseTensor *input_x = &x_in;
  const phi::DenseTensor *filter_x = &filter_x_in;
  const phi::DenseTensor *scale_x = &scale_x_in;
  const phi::DenseTensor *bias_x = &bias_x_in;
  // norm conv
  phi::DenseTensor *conv_out_x = conv_x;
  // sbar
  phi::DenseTensor *output = out;
  phi::DenseTensor *bitmask = bit_mask;
  // attrs
  double eps = static_cast<double>(epsilon);
  double momentum = static_cast<double>(momentum_in);

  bool is_train = !is_test && !use_global_stats;

  auto input_x_shape = common::vectorize<int>(input_x->dims());
  auto filter_x_shape = common::vectorize<int>(filter_x->dims());
  // std::swap used to convert shape of filter from conv2d when kernel size is
  // 1.
  if (filter_x_shape[1] != filter_x_shape[2] && 1 == filter_x_shape[2]) {
    std::swap(filter_x_shape[1], filter_x_shape[3]);
  }
  auto param_dims = scale_x->dims();
  auto param_shape = common::vectorize<int>(scale_x->dims());
  if (1 == param_shape.size()) {
    param_shape = {1, 1, 1, param_shape[0]};
  }
  auto output_shape = common::vectorize<int>(output->dims());
  auto bitmask_shape = common::vectorize<int>(bitmask->dims());
  int output_channel = filter_x_shape[0];
  int64_t ele_count =
      std::accumulate(
          output_shape.begin(), output_shape.end(), 1, std::multiplies<int>()) /
      output_channel;

  // 1. Conv
  phi::DenseTensor sum_x;
  phi::DenseTensor sum_of_squares_x;
  sum_x.Resize(param_dims);
  sum_of_squares_x.Resize(param_dims);
  phi::fusion::CudnnNormConvolution<T> conv_x_op(dev_ctx,
                                                 input_x_shape,
                                                 filter_x_shape,
                                                 output_shape,
                                                 padding,
                                                 stride,
                                                 dilation,
                                                 group);
  conv_x_op.Forward(
      dev_ctx, *input_x, *filter_x, conv_out_x, &sum_x, &sum_of_squares_x);

  // 2. BN
  phi::DenseTensor equiv_scale_x;
  phi::DenseTensor equiv_bias_x;
  equiv_scale_x.Resize(param_dims);
  equiv_bias_x.Resize(param_dims);
  phi::fusion::CudnnBNStatsFinalize<T> bn_x_op(dev_ctx, param_shape);
  bn_x_op.Forward(dev_ctx,
                  sum_x,
                  sum_of_squares_x,
                  *scale_x,
                  *bias_x,
                  saved_mean_x,
                  saved_invstd_x,
                  running_mean_x,
                  running_var_x,
                  &equiv_scale_x,
                  &equiv_bias_x,
                  eps,
                  momentum,
                  ele_count,
                  is_train);

  // 3. scale + bias + add + relu
  phi::fusion::CudnnScaleBiasAddRelu<T> sbar_op(dev_ctx,
                                                act_type,
                                                fuse_add,
                                                has_shortcut,
                                                output_shape,
                                                param_shape,
                                                bitmask_shape);
  if (has_shortcut) {
    // input z
    const phi::DenseTensor *input_z = z_in.get_ptr();
    const phi::DenseTensor *filter_z = filter_z_in.get_ptr();
    const phi::DenseTensor *scale_z = scale_z_in.get_ptr();
    const phi::DenseTensor *bias_z = bias_z_in.get_ptr();
    // norm conv
    phi::DenseTensor *conv_out_z = conv_z;

    auto input_z_shape = common::vectorize<int>(input_z->dims());
    auto filter_z_shape = common::vectorize<int>(filter_z->dims());

    // 3.1 Conv for second input
    phi::DenseTensor sum_z;
    phi::DenseTensor sum_of_squares_z;
    sum_z.Resize(param_dims);
    sum_of_squares_z.Resize(param_dims);
    phi::fusion::CudnnNormConvolution<T> conv_z_op(dev_ctx,
                                                   input_z_shape,
                                                   filter_z_shape,
                                                   output_shape,
                                                   padding,
                                                   stride_z,
                                                   dilation,
                                                   group);
    conv_z_op.Forward(
        dev_ctx, *input_z, *filter_z, conv_out_z, &sum_z, &sum_of_squares_z);

    // 3.2 BN for second input
    phi::DenseTensor equiv_scale_z;
    phi::DenseTensor equiv_bias_z;
    equiv_scale_z.Resize(param_dims);
    equiv_bias_z.Resize(param_dims);
    phi::fusion::CudnnBNStatsFinalize<T> bn_z_op(dev_ctx, param_shape);
    bn_z_op.Forward(dev_ctx,
                    sum_z,
                    sum_of_squares_z,
                    *scale_z,
                    *bias_z,
                    saved_mean_z,
                    saved_invstd_z,
                    running_mean_z,
                    running_var_z,
                    &equiv_scale_z,
                    &equiv_bias_z,
                    eps,
                    momentum,
                    ele_count,
                    is_train);
    // 3.3 sbar
    sbar_op.Forward(dev_ctx,
                    *conv_out_x,
                    equiv_scale_x,
                    equiv_bias_x,
                    conv_out_z,
                    &equiv_scale_z,
                    &equiv_bias_z,
                    output,
                    bitmask);
  } else {
    const phi::DenseTensor *input_z = fuse_add ? z_in.get_ptr() : nullptr;
    sbar_op.Forward(dev_ctx,
                    *conv_out_x,
                    equiv_scale_x,
                    equiv_bias_x,
                    input_z,
                    nullptr,
                    nullptr,
                    output,
                    bitmask);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(
    resnet_unit, GPU, ALL_LAYOUT, phi::ResNetUnitKernel, phi::dtype::float16) {}
#else
namespace phi {
template <typename T, typename Context>
void ResNetUnitEmptyKernel(const Context &dev_ctx,
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
  PADDLE_THROW(common::errors::Unavailable(
      "ResNetUnitOp only supports CUDNN_VERSION >= 8000 for now."));
}
}  // namespace phi
PD_REGISTER_KERNEL(resnet_unit,
                   GPU,
                   ALL_LAYOUT,
                   phi::ResNetUnitEmptyKernel,
                   phi::dtype::float16) {}
#endif
