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

#include "paddle/phi/kernels/conv_transpose_grad_kernel.h"
#include "paddle/phi/kernels/impl/conv_transpose_grad_kernel_impl.h"

#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/gpu/depthwise_conv.h"

namespace phi {

template <typename T, typename Context>
void Conv2dTransposeDoubleGradKernel(const Context& ctx,
                                     const DenseTensor& x,
                                     const DenseTensor& filter,
                                     const DenseTensor& dout,
                                     const DenseTensor& ddx,
                                     const DenseTensor& ddfilter,
                                     const std::vector<int>& strides,
                                     const std::vector<int>& paddings,
                                     const std::vector<int>& output_padding,
                                     const std::vector<int>& output_size,
                                     const std::string& padding_algorithm,
                                     int groups,
                                     const std::vector<int>& dilations,
                                     const std::string& data_format,
                                     DenseTensor* dx,
                                     DenseTensor* dfilter,
                                     DenseTensor* ddout) {
  ConvTransposeGradRawKernel<T, Context>(ctx,
                                         x,
                                         filter,
                                         dout,
                                         strides,
                                         paddings,
                                         padding_algorithm,
                                         groups,
                                         dilations,
                                         data_format,
                                         dx,
                                         dfilter);
}

template <typename T, typename Context>
void DepthwiseConv2dTransposeGradKernel(const Context& ctx,
                                        const DenseTensor& x,
                                        const DenseTensor& filter,
                                        const DenseTensor& dout,
                                        const std::vector<int>& strides,
                                        const std::vector<int>& paddings,
                                        const std::vector<int>& output_padding,
                                        const std::vector<int>& output_size,
                                        const std::string& padding_algorithm,
                                        int groups,
                                        const std::vector<int>& dilations,
                                        const std::string& data_format,
                                        DenseTensor* dx,
                                        DenseTensor* dfilter) {
  const DataLayout data_layout =
      paddle::framework::StringToDataLayout(data_format);
  DenseTensor filter_ = filter;

  if (!dx && !dfilter) {
    return;
  }

  std::vector<int> paddings_ = paddings;
  std::vector<int> dilations_ = dilations;

  auto x_dims = x.dims();
  auto filter_dims = filter_.dims();

  DDim in_data_dims;
  if (data_layout != DataLayout::kNHWC) {
    in_data_dims = slice_ddim(x_dims, 2, x_dims.size());
  } else {
    in_data_dims = slice_ddim(x_dims, 1, x_dims.size() - 1);
  }
  DDim filter_data_dims = slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings_, &dilations_, padding_algorithm, in_data_dims, strides, ksize);

  if (dx) {
    paddle::operators::math::DepthwiseConvFunctor<Context, T> depthwiseConv;
    depthwiseConv(ctx,
                  dout,
                  filter_,
                  strides,
                  std::vector<int>{
                      paddings_[0], paddings_[2], paddings_[1], paddings_[3]},
                  dilations_,
                  dx,
                  data_layout);
  }

  if (dfilter) {
    funcs::SetConstant<Context, T> set_zero;
    ctx.template Alloc<T>(dfilter);
    set_zero(ctx, dfilter, static_cast<T>(0));

    paddle::operators::math::DepthwiseConvFilterGradFunctor<Context, T>
        depthwiseConvFilterGrad;
    depthwiseConvFilterGrad(
        ctx,
        dout,
        x,
        strides,
        std::vector<int>{
            paddings_[0], paddings_[2], paddings_[1], paddings_[3]},
        dilations_,
        dfilter,
        data_layout);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(conv2d_transpose_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::Conv2dTransposeGradKernel,
                   float,
                   double) {}
PD_REGISTER_KERNEL(conv2d_transpose_grad_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::Conv2dTransposeDoubleGradKernel,
                   float,
                   double) {}
PD_REGISTER_KERNEL(conv3d_transpose_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::Conv3dTransposeGradKernel,
                   float,
                   double) {}
PD_REGISTER_KERNEL(depthwise_conv2d_transpose_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::DepthwiseConv2dTransposeGradKernel,
                   float,
                   double) {}
