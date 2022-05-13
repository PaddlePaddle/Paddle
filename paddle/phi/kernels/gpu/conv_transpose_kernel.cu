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

#include "paddle/phi/kernels/conv_transpose_kernel.h"
#include "paddle/phi/kernels/impl/conv_transpose_kernel_impl.h"

#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/gpu/depthwise_conv.h"

namespace phi {

template <typename T, typename Context>
void DepthwiseConv2dTransposeKernel(const Context& ctx,
                                    const DenseTensor& x,
                                    const DenseTensor& filter,
                                    const std::vector<int>& strides,
                                    const std::vector<int>& paddings,
                                    const std::vector<int>& output_padding,
                                    const std::vector<int>& output_size,
                                    const std::string& padding_algorithm,
                                    int groups,
                                    const std::vector<int>& dilations,
                                    const std::string& data_format,
                                    DenseTensor* out) {
  const DataLayout data_layout =
      paddle::framework::StringToDataLayout(data_format);
  DenseTensor filter_ = filter;
  ctx.template Alloc<T>(out);

  PADDLE_ENFORCE_EQ(
      groups,
      filter_.dims()[0],
      errors::InvalidArgument(
          "groups should be error to the 1st dimension of filter_. But "
          "received groups is %d and filter dimension[0] is %d",
          groups,
          filter_.dims()[0]));

  std::vector<int> paddings_ = paddings;
  std::vector<int> dilations_ = dilations;

  for (auto v : dilations_) {
    PADDLE_ENFORCE_EQ(
        v,
        1,
        errors::InvalidArgument("dilations should be 1 in depthwise conv. "
                                "But received dilations is %d",
                                v));
  }

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

  ctx.template Alloc<T>(out);

  funcs::SetConstant<Context, T> set_zero;
  set_zero(ctx, out, static_cast<T>(0));

  paddle::operators::math::DepthwiseConvInputGradFunctor<Context, T>
      depthwiseConvInputGrad;
  depthwiseConvInputGrad(
      ctx,
      *out,
      filter,
      x,
      strides,
      std::vector<int>{paddings_[0], paddings_[2], paddings_[1], paddings_[3]},
      dilations_,
      out,
      data_layout);
}

}  // namespace phi

PD_REGISTER_KERNEL(conv2d_transpose,
                   GPU,
                   ALL_LAYOUT,
                   phi::Conv2dTransposeKernel,
                   float,
                   double) {}
PD_REGISTER_KERNEL(conv3d_transpose,
                   GPU,
                   ALL_LAYOUT,
                   phi::Conv3dTransposeKernel,
                   float,
                   double) {}
PD_REGISTER_KERNEL(depthwise_conv2d_transpose,
                   GPU,
                   ALL_LAYOUT,
                   phi::DepthwiseConv2dTransposeKernel,
                   float,
                   double) {}
