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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/funcs/batch_norm_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/gpu/depthwise_conv.h"

namespace phi {

template <typename T, typename Context>
void DepthwiseConvGradKernel(const Context& dev_ctx,
                             const DenseTensor& input,
                             const DenseTensor& filter,
                             const DenseTensor& out_grad,
                             const std::vector<int>& strides_t,
                             const std::vector<int>& paddings_t,
                             const std::string& padding_algorithm,
                             int groups,
                             const std::vector<int>& dilations_t,
                             const std::string& data_format,
                             DenseTensor* input_grad,
                             DenseTensor* filter_grad) {
  const DenseTensor* output_grad = &out_grad;

  if (!input_grad && !filter_grad) return;

  bool has_fuse_relu = dev_ctx.HasDnnAttr("fuse_relu_before_depthwise_conv");
  bool fuse_relu =
      has_fuse_relu
          ? PADDLE_GET_CONST(
                bool, dev_ctx.GetDnnAttr("fuse_relu_before_depthwise_conv"))
          : false;

  std::vector<int> strides = strides_t;
  std::vector<int> paddings = paddings_t;
  std::vector<int> dilations = dilations_t;

  // update padding and dilation
  auto in_dims = input.dims();
  auto filter_dims = filter.dims();

  DDim in_data_dims;
  const phi::DataLayout data_layout = phi::StringToDataLayout(data_format);
  if (data_layout != phi::DataLayout::kNHWC) {
    in_data_dims = slice_ddim(in_dims, 2, in_dims.size());
  } else {
    in_data_dims = slice_ddim(in_dims, 1, in_dims.size() - 1);
  }
  DDim filter_data_dims = slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  bool is_sys_pad = strides.size() * 2 == paddings.size() ? false : true;
  if (!is_sys_pad) {
    for (size_t i = 0; i < strides.size(); ++i) {
      paddings.erase(paddings.begin() + i + 1);
    }
  }
  phi::funcs::SetConstant<Context, T> set_zero;

  if (input_grad) {
    dev_ctx.template Alloc<T>(input_grad);
    set_zero(dev_ctx, input_grad, static_cast<T>(0));

    if (fuse_relu) {
      paddle::operators::math::DepthwiseConvInputGradFunctor<Context, T, true>
          depthwiseConvInputGrad;
      depthwiseConvInputGrad(dev_ctx,
                             input,
                             filter,
                             *output_grad,
                             strides,
                             paddings,
                             dilations,
                             input_grad,
                             data_layout);
    } else {
      paddle::operators::math::DepthwiseConvInputGradFunctor<Context, T, false>
          depthwiseConvInputGrad;
      depthwiseConvInputGrad(dev_ctx,
                             input,
                             filter,
                             *output_grad,
                             strides,
                             paddings,
                             dilations,
                             input_grad,
                             data_layout);
    }
  }

  if (filter_grad) {
    dev_ctx.template Alloc<T>(filter_grad);
    set_zero(dev_ctx, filter_grad, static_cast<T>(0));
    if (fuse_relu) {
      paddle::operators::math::DepthwiseConvFilterGradFunctor<Context, T, true>
          depthwiseConvFilterGrad;
      depthwiseConvFilterGrad(dev_ctx,
                              input,
                              *output_grad,
                              strides,
                              paddings,
                              dilations,
                              filter_grad,
                              data_layout);
    } else {
      paddle::operators::math::DepthwiseConvFilterGradFunctor<Context, T, false>
          depthwiseConvFilterGrad;
      depthwiseConvFilterGrad(dev_ctx,
                              input,
                              *output_grad,
                              strides,
                              paddings,
                              dilations,
                              filter_grad,
                              data_layout);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(depthwise_conv2d_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::DepthwiseConvGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
