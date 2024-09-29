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

#include <glog/logging.h>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/fusion/cutlass/conv2d/conv2d_decl.h"

#include "paddle/phi/backends/dynload/cutlass_conv2d.h"

namespace phi {
namespace fusion {
namespace cutlass_internal {

typedef bool (*func)(phi::fusion::cutlass_internal::ConvAllParams);

template <typename T, typename Context>
void FusedConv2dAddActKernel(const Context& ctx,
                             const DenseTensor& x,
                             const DenseTensor& filter,
                             const DenseTensor& bias,
                             const paddle::optional<DenseTensor>& residual,
                             const std::vector<int>& strides,
                             const std::vector<int>& paddings,
                             const std::string& padding_algorithm,
                             const std::vector<int>& dilations,
                             int groups,
                             const std::string& data_format,
                             const std::string& activation,
                             const std::vector<int>& split_channels,
                             bool exhaustive_search,
                             int workspace_size_MB,
                             float fuse_alpha,
                             DenseTensor* output,
                             std::vector<DenseTensor*> outputs) {
  ctx.template Alloc<T>(output);
  auto in_dims = x.dims();
  auto filter_dims = filter.dims();
  auto out_dims = output->dims();
  PADDLE_ENFORCE_EQ(
      in_dims.size(),
      4UL,
      common::errors::InvalidArgument(
          "The input tensor X's dimensions should be 4, but got %d.",
          in_dims.size()));
  PADDLE_ENFORCE_EQ(
      filter_dims.size(),
      4UL,
      common::errors::InvalidArgument(
          "The input tensor filter's dimensions must be 4, but got %d.",
          filter_dims.size()));
  PADDLE_ENFORCE_EQ(
      strides.size(),
      2UL,
      common::errors::InvalidArgument(
          "The size of strides must be 2, but got %d.", strides.size()));
  PADDLE_ENFORCE_EQ(
      dilations.size(),
      2UL,
      common::errors::InvalidArgument(
          "The size of dilations must be 2, but got %d.", dilations.size()));

  PADDLE_ENFORCE_EQ(padding_algorithm,
                    "EXPLICIT",
                    common::errors::InvalidArgument(
                        "The padding_algorithm must be EXPLICIT, but got %s.",
                        padding_algorithm));
  PADDLE_ENFORCE_EQ(
      data_format,
      "NHWC",
      common::errors::InvalidArgument(
          "The data_format must be NHWC, but got %s.", data_format));
  const int batch = in_dims[0];
  const int ic = in_dims[3];
  const int ih = in_dims[1];
  const int iw = in_dims[2];

  PADDLE_ENFORCE_EQ(
      ic,
      groups * filter_dims[3],
      common::errors::InvalidArgument(
          "The last dimension of X (%d) must be equal to "
          "groups (%d) multiply the last dimension of filter (%d).",
          ic,
          groups,
          filter_dims[3]));
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
    PADDLE_THROW(common::errors::InvalidArgument(
        "Attr paddins in fused_conv2d_add_act "
        "must have 2 or 4 elements, but now have "
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

  PADDLE_ENFORCE_EQ(
      out_dims.size(),
      4UL,
      common::errors::InvalidArgument(
          "The output's dimensions must be 4, but got %d.", out_dims.size()));
  const int oh = out_dims[1];
  const int ow = out_dims[2];

  int64_t device_id = ctx.GetPlace().GetDeviceId();
  int sm_version = backends::gpu::GetGPUComputeCapability(device_id);

  auto get_conv2d_dtype = [&](decltype(x.dtype()) x_type)
      -> phi::fusion::cutlass_internal::Conv2dDataType {
    switch (x_type) {
      case phi::DataType::FLOAT32:
        return Conv2dDataType::fp32;
      case phi::DataType::FLOAT16:
        return Conv2dDataType::fp16;
      case phi::DataType::BFLOAT16:
        return Conv2dDataType::bf16;
    }
  };

  auto cutlass_dispatch_sm_version = [&](int device_sm_version) -> int {
    if (device_sm_version < 75) {
      PADDLE_ENFORCE_GE(
          device_sm_version,
          75,
          common::errors::PreconditionNotMet(
              "fused_conv2d_add_act only supports sm >= 75, but got %d.",
              device_sm_version));
    } else if (device_sm_version > 80) {
      return 80;
    } else {
      return device_sm_version;
    }
  };

  ConvAllParams params = {
      reinterpret_cast<const void*>(x.data<T>()),
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
      ctx.stream(),
      0,  // alpha
      cutlass_dispatch_sm_version(sm_version),
      get_conv2d_dtype(x.dtype()),
      nullptr,
  };

  void* dlhandler = phi::dynload::GetCutlassConv2dHandle();
  func conv_func = NULL;
  PADDLE_ENFORCE_NOT_NULL(
      dlhandler,
      common::errors::NotFound("Fail to get CutlassConv2d handler."));

  // conv2d_depthwise
  if (groups == ic && ic == oc) {
    // conv2d_depthwise need a tmp workspace.
    phi::Allocator::AllocationPtr tmp_ptr = phi::memory_utils::Alloc(
        ctx.GetPlace(),
        oc * kh * kw * sizeof(T),
        phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
    params.workspace = tmp_ptr->ptr();
    // cutlass conv2d_depthwise not support residual
    if (residual) {
      PADDLE_ENFORCE_EQ(residual->data<T>(),
                        nullptr,
                        common::errors::InvalidArgument(
                            "The pointer of residual's data must be null."));
    }
    if (activation == "relu") {
      conv_func = (func)(dlsym(dlhandler, "Conv2dDepthwiseBiasRelu"));
    } else if (activation == "identity") {
      conv_func = (func)(dlsym(dlhandler, "Conv2dDepthwiseBias"));
    } else if (activation == "sigmoid") {
      conv_func = (func)(dlsym(dlhandler, "Conv2dDepthwiseBiasSigmoid"));
    } else if (activation == "swish") {
      conv_func = (func)(dlsym(dlhandler, "Conv2dDepthwiseBiasSilu"));
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Cutlass conv2d_depthwise does not support this activation: %s.",
          activation.c_str()));
    }

    if (!conv_func(params)) {
      PADDLE_THROW(
          common::errors::Fatal("no fused_conv2d_add_act cutlass kernel "));
    }

    output->set_layout(DataLayout::NHWC);
    return;
  }

  // below: fused_conv2d_add_act && groups == 1
  PADDLE_ENFORCE_EQ(groups,
                    1,
                    common::errors::InvalidArgument(
                        "The groups must be 1, but got %d.", groups));
  if (residual) {
    if (activation == "relu") {
      params.residual = reinterpret_cast<const void*>(residual->data<T>());
      conv_func = (func)(dlsym(dlhandler, "Conv2dBiasAddRelu"));
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Cutlass now only support relu activation in a residual block, but "
          "got %s.",
          activation.c_str()));
    }
  } else if (activation == "relu") {
    conv_func = (func)(dlsym(dlhandler, "Conv2dBiasRelu"));
  } else if (activation == "swish") {
    conv_func = (func)(dlsym(dlhandler, "Conv2dBiasSilu"));
  } else if (activation == "identity") {
    conv_func = (func)(dlsym(dlhandler, "Conv2dBias"));
  } else if (activation == "leaky_relu") {
    conv_func = (func)(dlsym(dlhandler, "Conv2dBiasLeakyRelu"));
    params.alpha = fuse_alpha;
  } else if (activation == "sigmoid") {
    conv_func = (func)(dlsym(dlhandler, "Conv2dBiasSigmoid"));
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Cutlass does not support this activation: %s.", activation.c_str()));
  }

  if (!conv_func(params)) {
    PADDLE_THROW(
        common::errors::Fatal("no fused_conv2d_add_act cutlass kernel "));
  }

  output->set_layout(DataLayout::NHWC);
}
}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_conv2d_add_act,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::cutlass_internal::FusedConv2dAddActKernel,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
