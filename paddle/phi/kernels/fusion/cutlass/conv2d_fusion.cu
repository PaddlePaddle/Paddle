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

  CHECK_EQ(padding_algorithm == "EXPLICIT", true);
  const int batch = in_dims[0];
  const int ic = in_dims[3];
  const int ih = in_dims[1];
  const int iw = in_dims[2];

  CHECK_EQ(ic == groups * filter_dims[3], true);
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
    PADDLE_THROW(phi::errors::InvalidArgument(
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

  auto dtype2string = [&](decltype(x.dtype()) x_type) -> std::string {
    switch (x_type) {
      case phi::DataType::FLOAT32:
        return "fp32";
      case phi::DataType::FLOAT16:
        return "fp16";
      case phi::DataType::BFLOAT16:
        return "bf16";
    }
    return "";
  };
  ConvAllParams params = {reinterpret_cast<const void*>(x.data<T>()),
                          reinterpret_cast<const void*>(filter.data<T>()),
                          reinterpret_cast<const void*>(bias.data<T>()),
                          nullptr,
                          reinterpret_cast<void*>(output->data<T>()),
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
                          groups,
                          &ctx,
                          0,   // alpha
                          80,  // sm_version
                          dtype2string(x.dtype())};

  // conv2d_depthwise
  if (groups == ic && ic == oc) {
    // cutlass conv2d_depthwise not support residual
    if (residual) {
      CHECK_EQ(residual->data<T>() == nullptr, true);
    }
    if (activation == "relu") {
      Conv2dDepthwiseBiasRelu(params);
    } else if (activation == "identity") {
      Conv2dDepthwiseBias(params);
    } else if (activation == "sigmoid") {
      Conv2dDepthwiseBiasSigmoid(params);
    } else if (activation == "swish") {
      Conv2dDepthwiseBiasSilu(params);
    } else {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Cutlass conv2d_depthwise does not support this activation: %s.",
          activation.c_str()));
    }
    return;
  }

  // below: conv2d_fusion && groups == 1
  CHECK_EQ(groups == 1, true);
  if (residual) {
    params.residual = reinterpret_cast<const void*>(residual->data<T>());
    if (activation == "relu") {
      Conv2dBiasAddRelu(params);
    } else if (activation == "identity_elementwise_add_identity") {
      Conv2dBiasAdd(params);
    } else if (activation == "swish_elementwise_add_identity") {
      Conv2dBiasSiluAdd(params);
    } else {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Cutlass now only support %s in a residual block",
          activation.c_str()));
    }
  } else if (activation == "relu") {
    Conv2dBiasRelu(params);
  } else if (activation == "swish") {
    Conv2dBiasSilu(params);
  } else if (activation == "identity") {
    Conv2dBias(params);
  } else if (activation == "leaky_relu") {
    params.alpha = fuse_alpha;
    Conv2dBiasLeakyRelu(params);
  } else if (activation == "sigmoid") {
    Conv2dBiasSigmoid(params);
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
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
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
