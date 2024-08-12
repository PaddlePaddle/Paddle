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
void ResNetUnitGradKernel(
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
  PADDLE_ENFORCE_EQ(phi::backends::gpu::CudnnDataType<T>::type,
                    CUDNN_DATA_HALF,
                    common::errors::Unavailable(
                        "ResNetUnitOp only supports float16 for now."));

  const phi::DenseTensor *y_grad = &out_grad;

  const phi::DenseTensor *x = &x_in;
  const phi::DenseTensor *filter_x = &filter_x_in;
  const phi::DenseTensor *scale_x = &scale_x_in;
  const phi::DenseTensor *bias_x = &bias_x_in;
  const phi::DenseTensor *saved_mean_x = &saved_mean_x_in;
  const phi::DenseTensor *saved_invstd_x = &saved_invstd_x_in;

  const phi::DenseTensor *conv_out_x = &conv_x_in;
  const phi::DenseTensor *output = &out;
  const phi::DenseTensor *bitmask = &bit_mask;

  double eps = static_cast<double>(epsilon);
  double momentum = static_cast<double>(momentum_in);

  auto x_shape = common::vectorize<int>(x->dims());
  auto filter_x_shape = common::vectorize<int>(filter_x->dims());
  auto param_shape = common::vectorize<int>(scale_x->dims());
  auto output_shape = common::vectorize<int>(output->dims());
  auto bitmask_shape = common::vectorize<int>(bitmask->dims());

  // 1. Backward of BN (+ Add + Relu) for x, get conv_out_x_grad,
  // scale_x_grad, bias_x_grad
  phi::DenseTensor conv_out_x_grad;
  conv_out_x_grad.Resize(conv_out_x->dims());
  phi::fusion::CudnnScaleBiasAddRelu<T> sbar_x_op(dev_ctx,
                                                  act_type,
                                                  fuse_add,
                                                  has_shortcut,
                                                  output_shape,
                                                  param_shape,
                                                  bitmask_shape);
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
    const phi::DenseTensor *bias_z = bias_z_in.get_ptr();
    const phi::DenseTensor *saved_mean_z = saved_mean_z_in.get_ptr();
    const phi::DenseTensor *saved_invstd_z = saved_invstd_z_in.get_ptr();
    const phi::DenseTensor *conv_out_z = conv_z_in.get_ptr();

    // 1.1 Backward of BN + Add (+ Relu) for x, get conv_out_x_grad,
    // scale_x_grad, bias_x_grad and z_grad_temp
    phi::DenseTensor z_grad_temp;
    z_grad_temp.Resize(conv_out_z->dims());
    sbar_x_op.Backward(dev_ctx,
                       *y_grad,
                       *conv_out_x,
                       *scale_x,
                       *bias_x,
                       *saved_mean_x,
                       *saved_invstd_x,
                       bitmask,
                       &conv_out_x_grad,
                       &z_grad_temp,
                       scale_x_grad,
                       bias_x_grad,
                       eps);

    // 1.2 bn backward for z, get conv_out_z_grad, dscale_z, dbias_z
    phi::DenseTensor conv_out_z_grad;
    conv_out_z_grad.Resize(conv_out_z->dims());
    phi::fusion::CudnnScaleBiasAddRelu<T> sbar_z_op(
        dev_ctx, "", false, false, output_shape, param_shape, bitmask_shape);
    sbar_z_op.Backward(dev_ctx,
                       z_grad_temp,
                       *conv_out_z,
                       *scale_z,
                       *bias_z,
                       *saved_mean_z,
                       *saved_invstd_z,
                       nullptr,
                       &conv_out_z_grad,
                       nullptr,
                       scale_z_grad,
                       bias_z_grad,
                       eps);

    // 1.3 Backward of Conv for z, get z_grad and filter_z_grad
    auto z_shape = common::vectorize<int>(z->dims());
    auto filter_z_shape = common::vectorize<int>(filter_z->dims());
    phi::fusion::CudnnNormConvolutionGrad<T> conv_z_op(dev_ctx,
                                                       z_shape,
                                                       filter_z_shape,
                                                       output_shape,
                                                       padding,
                                                       stride_z,
                                                       dilation,
                                                       group);
    conv_z_op.Backward(
        dev_ctx, *z, *filter_z, conv_out_z_grad, z_grad, filter_z_grad);
  } else {
    // 1.1 Backward of BN (+ Add + Relu) for x, get conv_out_x_grad,
    // scale_x_grad, bias_x_grad (and z_grad)
    phi::DenseTensor *z_grad_tmp = fuse_add ? z_grad : nullptr;
    sbar_x_op.Backward(dev_ctx,
                       *y_grad,
                       *conv_out_x,
                       *scale_x,
                       *bias_x,
                       *saved_mean_x,
                       *saved_invstd_x,
                       bitmask,
                       &conv_out_x_grad,
                       z_grad_tmp,
                       scale_x_grad,
                       bias_x_grad,
                       eps);
  }

  // 2. Backward of Conv for x, get x_grad and filter_x_grad
  phi::fusion::CudnnNormConvolutionGrad<T> conv_x_op(dev_ctx,
                                                     x_shape,
                                                     filter_x_shape,
                                                     output_shape,
                                                     padding,
                                                     stride,
                                                     dilation,
                                                     group);
  conv_x_op.Backward(dev_ctx,
                     *x,
                     *filter_x,
                     conv_out_x_grad,
                     x_grad,
                     filter_x_grad,
                     use_addto);
}

}  // namespace phi

PD_REGISTER_KERNEL(resnet_unit_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ResNetUnitGradKernel,
                   phi::dtype::float16) {}
#else
namespace phi {

template <typename T, typename Context>
void ResNetUnitGradEmptyKernel(
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
    DenseTensor *bias_z_grad) {}
}  // namespace phi

PD_REGISTER_KERNEL(resnet_unit_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ResNetUnitGradEmptyKernel,
                   phi::dtype::float16) {}
#endif
