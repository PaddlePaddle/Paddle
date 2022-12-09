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
#include <algorithm>
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/kernels/fusion/cutlass/conv2d/conv2d_all.h"

namespace phi {
namespace fusion {

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
  int batch = in_dims[0];
  int ic = in_dims[3];
  int ih = in_dims[1];
  int iw = in_dims[2];
  int pad_h = paddings[0];
  int pad_w = paddings[1];
  int stride_h = strides[0];
  int stride_w = strides[1];
  int oc = filter_dims[0];
  int kh = filter_dims[1];
  int kw = filter_dims[2];

  int oh = (ih + pad_h * 2 - kh) / stride_h + 1;
  int ow = (iw + pad_w * 2 - kw) / stride_w + 1;

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
                          pad_h,
                          pad_w,
                          stride_h,
                          stride_w,
                          ctx.stream()};

  if (residual) {
    if (activation == "relu") {
      params.residual = reinterpret_cast<const half*>(residual->data<T>());
      cutlass_conv2d_bias_add_relu(params);
    } else {
      PADDLE_THROW(paddle::platform::errors::InvalidArgument(
          "Cutlass now only support relu activation in a residual block"));
    }
  } else if (activation == "relu") {
    cutlass_conv2d_bias_relu(params);
  } else if (activation == "swish") {
    cutlass_conv2d_bias_silu(params);
  } else if (activation == "identity") {
    cutlass_conv2d_bias(params);
  } else if (activation == "leaky_relu") {
    params.alpha = fuse_alpha;
    cutlass_conv2d_bias_leaky_relu(params);
  } else {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "Cutlass does not support this activation: %s.", activation.c_str()));
  }
  output->set_layout(DataLayout::NHWC);
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(conv2d_fusion_cutlass,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::Conv2dFusionKernel,
                   phi::dtype::float16) {}
