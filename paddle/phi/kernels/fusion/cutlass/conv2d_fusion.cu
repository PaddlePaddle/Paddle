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

#include "paddle/phi/kernels/fusion/conv2d_fusion.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/kernels/fusion/cutlass/conv2d/conv2d_decl.h"

namespace phi {
namespace fusion {
namespace cutlass_internal {
template <typename T, typename Context>
void Conv2dFusionKernel(const Context& ctx,
                        const DenseTensor& x,
                        const DenseTensor& filter,
                        const DenseTensor& bias,
                        const paddle::optional<DenseTensor>& residual,
                        const std::vector<int>& strides,
                        const std::vector<int>& paddings,
                        const std::string& padding_algorithm,
                        int groups,
                        const std::vector<int>& dilations,
                        const std::string& data_format,
                        const std::string& activation,
                        float fuse_alpha,
                        DenseTensor* output) {
  ctx.template Alloc<T>(output);
  auto in_dims = x.dims();
  auto filter_dims = filter.dims();
  auto out_dims = output->dims();
  CHECK_EQ(in_dims.size() == 4UL, true);
  CHECK_EQ(filter_dims.size() == 4UL, true);
  CHECK_EQ(strides.size() == 2UL, true);
  CHECK_EQ(dilations.size() == 2UL, true);
  const int batch = in_dims[0];
  const int ic = in_dims[3];
  const int ih = in_dims[1];
  const int iw = in_dims[2];
  int pad_h0 = 0;
  int pad_h1 = 0;
  int pad_w0 = 0;
  int pad_w1 = 0;
  if (paddings.size() == 2UL) {
    pad_h0 = paddings[0];
    pad_h1 = paddings[0];
    pad_w0 = paddings[1];
    pad_w1 = paddings[1];
  } else if (paddings.size() == 4UL) {
    pad_h0 = paddings[0];
    pad_h1 = paddings[1];
    pad_w0 = paddings[2];
    pad_w1 = paddings[3];
  } else {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "Attr paddins in conv2d_fusion must have 2 or 4 elements, but now have "
        "%u elements.",
        paddings.size()));
  }

  const int stride_h = strides[0];
  const int stride_w = strides[1];
  const int dilation_h = dilations[0];
  const int dilation_w = dilations[1];
  const int oc = filter_dims[0];
  const int kh = filter_dims[1];
  const int kw = filter_dims[2];

  CHECK_EQ(out_dims.size() == 4UL, true);
  const int oh = out_dims[1];
  const int ow = out_dims[2];

  ConvAllParams params = {reinterpret_cast<const half*>(x.data<T>()),
                          reinterpret_cast<const half*>(filter.data<T>()),
                          reinterpret_cast<const half*>(bias.data<T>()),
                          nullptr,
                          reinterpret_cast<half*>(output->data<T>()),
                          batch,
                          ic,
                          ih,
                          iw,
                          kh,
                          kw,
                          oc,
                          pad_h0,
                          pad_h1,
                          pad_w0,
                          pad_w1,
                          stride_h,
                          stride_w,
                          dilation_h,
                          dilation_w,
                          oh,
                          ow,
                          &ctx};

  if (residual) {
    if (activation == "relu") {
      params.residual = reinterpret_cast<const half*>(residual->data<T>());
      conv2d_bias_add_relu(params);
    } else {
      PADDLE_THROW(paddle::platform::errors::InvalidArgument(
          "Cutlass now only support relu activation in a residual block"));
    }
  } else if (activation == "relu") {
    conv2d_bias_relu(params);
  } else if (activation == "swish") {
    conv2d_bias_silu(params);
  } else if (activation == "identity") {
    conv2d_bias(params);
  } else if (activation == "leaky_relu") {
    params.alpha = fuse_alpha;
    conv2d_bias_leaky_relu(params);
  } else {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "Cutlass does not support this activation: %s.", activation.c_str()));
  }
  output->set_layout(DataLayout::NHWC);
}
}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(conv2d_fusion_cutlass,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::cutlass_internal::Conv2dFusionKernel,
                   phi::dtype::float16) {}
