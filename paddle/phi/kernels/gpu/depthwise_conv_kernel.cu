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
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/funcs/batch_norm_utils.h"
#include "paddle/phi/kernels/gpu/depthwise_conv.h"

namespace phi {

template <typename T, typename Context>
void DepthwiseConvKernel(const Context& dev_ctx,
                         const DenseTensor& input,
                         const DenseTensor& filter,
                         const std::vector<int>& strides_t,
                         const std::vector<int>& paddings_t,
                         const std::string& padding_algorithm,
                         int groups,
                         const std::vector<int>& dilations_t,
                         const std::string& data_format,
                         DenseTensor* out) {
  DenseTensor* output = out;
  dev_ctx.template Alloc<T>(output);

  const std::vector<int> strides = strides_t;
  std::vector<int> dilations = dilations_t;
  std::vector<int> paddings = paddings_t;

  const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

  bool has_fuse_relu = dev_ctx.HasDnnAttr("fuse_relu_before_depthwise_conv");
  bool fuse_relu =
      has_fuse_relu
          ? PADDLE_GET_CONST(
                bool, dev_ctx.GetDnnAttr("fuse_relu_before_depthwise_conv"))
          : false;

  if (channel_last) {
    PADDLE_ENFORCE_EQ(
        output->dims()[output->dims().size() - 1] %
            input.dims()[input.dims().size() - 1],
        0,
        common::errors::InvalidArgument(
            "ShapeError: The output channels must be a multiple of the "
            "input channels. But receivced output channel number is %d "
            "and input channel number is %d",
            output->dims()[output->dims().size() - 1],
            input.dims()[input.dims().size() - 1]));
  } else {
    PADDLE_ENFORCE_EQ(
        output->dims()[1] % input.dims()[1],
        0,
        common::errors::InvalidArgument(
            "ShapeError: The output channels must be a multiple of the "
            "input channels. But receivced output channel number is %d "
            "and input channel number is %d",
            output->dims()[1],
            input.dims()[1]));
  }

  // update padding and dilation
  auto in_dims = input.dims();
  auto filter_dims = filter.dims();

  DDim in_data_dims;
  const phi::DataLayout data_layout = common::StringToDataLayout(data_format);
  if (data_layout != phi::DataLayout::kNHWC) {
    in_data_dims = slice_ddim(in_dims, 2, in_dims.size());
  } else {
    in_data_dims = slice_ddim(in_dims, 1, in_dims.size() - 1);
  }

  DDim filter_data_dims = slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = common::vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  bool is_sys_pad = strides.size() * 2 == paddings.size() ? false : true;
  if (!is_sys_pad) {
    for (size_t i = 0; i < strides.size(); ++i) {
      paddings.erase(paddings.begin() + i + 1);
    }
  }

  if (fuse_relu) {
    paddle::operators::math::DepthwiseConvFunctor<Context, T, true>
        depthwiseConv;
    depthwiseConv(dev_ctx,
                  input,
                  filter,
                  strides,
                  paddings,
                  dilations,
                  output,
                  data_layout);
  } else {
    paddle::operators::math::DepthwiseConvFunctor<Context, T, false>
        depthwiseConv;
    depthwiseConv(dev_ctx,
                  input,
                  filter,
                  strides,
                  paddings,
                  dilations,
                  output,
                  data_layout);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(depthwise_conv2d,
                   GPU,
                   ALL_LAYOUT,
                   phi::DepthwiseConvKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
