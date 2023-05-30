/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/conv_op.h"

#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/operators/generator/get_expected_kernel_func.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif
#include "paddle/fluid/framework/infershape_utils.h"

namespace paddle {
namespace operators {

std::vector<int64_t> ConvOp::ComputeOutputShape(
    framework::InferShapeContext* ctx) const {
  OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "Conv");
  OP_INOUT_CHECK(ctx->HasInput("Filter"), "Input", "Filter", "Conv");

  auto in_dims = ctx->GetInputDim("Input");
  auto filter_dims = ctx->GetInputDim("Filter");

  std::vector<int> strides = ctx->Attrs().Get<std::vector<int>>("strides");
  std::vector<int> paddings = ctx->Attrs().Get<std::vector<int>>("paddings");
  std::string padding_algorithm =
      ctx->Attrs().Get<std::string>("padding_algorithm");
  int groups = ctx->Attrs().Get<int>("groups");
  std::vector<int> dilations = ctx->Attrs().Get<std::vector<int>>("dilations");
  int dilation_size = dilations.size();
  for (int i = 0; i < dilation_size; ++i) {
    PADDLE_ENFORCE_GT(
        dilations[i],
        0,
        platform::errors::InvalidArgument(
            "The dilation of Op(Conv) should be larget than 0, but received "
            "dilation is %d.",
            dilations[i]));
  }
  const std::string data_format = ctx->Attrs().Get<std::string>("data_format");

  // MKL-DNN Kernels are using NCHW order of dims description
  // so we ignore data_format consideration for MKL-DNN kernel
  const bool channel_last = (ctx->IsRunMKLDNNKernel() == false) &&
                            (data_format == "NHWC" || data_format == "NDHWC");

  PADDLE_ENFORCE_EQ(
      in_dims.size() == 4 || in_dims.size() == 5,
      true,
      platform::errors::InvalidArgument(
          "The input of Op(Conv) should be a 4-D or 5-D Tensor. But "
          "received: input's dimension is %u, input's shape is [%s].",
          in_dims.size(),
          in_dims));

  PADDLE_ENFORCE_EQ(
      in_dims.size(),
      filter_dims.size(),
      platform::errors::InvalidArgument(
          "The input's dimension and filter's dimension of "
          "Op(Conv) should be equal. But received: the input's shape is [%s], "
          "the input's dimension is %d; the filter's shape is [%s],  "
          "the filter's dimension is %d.",
          in_dims,
          in_dims.size(),
          filter_dims,
          filter_dims.size()));

  int stride_size = strides.size();
  for (int i = 0; i < stride_size; ++i) {
    PADDLE_ENFORCE_GT(
        strides[i],
        0,
        platform::errors::InvalidArgument(
            "The stride of Op(Conv) should be larget than 0, but received "
            "stride is %d.",
            strides[i]));
  }

  int in_sub_stride_size = in_dims.size() - stride_size;
  PADDLE_ENFORCE_EQ(
      in_dims.size(),
      strides.size() + 2U,
      platform::errors::InvalidArgument(
          "The difference of input's dimension and Attr(strides)'s "
          "length must be euqal to 2 for Op(Conv). "
          "But received: input's dimension is %d, input's shape is [%s]; "
          "Attr(stride)'s length is %d, Attr(stride) is [%s]; "
          "difference of input's dimention and Attr(strides)'s length = %u.",
          in_dims.size(),
          in_dims,
          strides.size(),
          phi::make_ddim(strides),
          in_sub_stride_size));

  const auto input_channels =
      channel_last ? in_dims[in_dims.size() - 1] : in_dims[1];

  PADDLE_ENFORCE_EQ(
      input_channels,
      filter_dims[1] * groups,
      platform::errors::InvalidArgument(
          "The number of input's channels should be equal to filter's channels "
          "* groups for Op(Conv). But received: the input's channels is %d, "
          "the input's shape is [%s]; the filter's channels is %d, the "
          "filter's shape is [%s]; the groups is %d, the data_format is %s. "
          "The error may come from wrong data_format setting.",
          input_channels,
          in_dims,
          filter_dims[1],
          filter_dims,
          groups,
          data_format));
  PADDLE_ENFORCE_EQ(
      filter_dims[0] % groups,
      0,
      platform::errors::InvalidArgument(
          "The number of output's channels (filter's first dimension) of "
          "Op(Conv) should be divided by groups. But received: "
          "the output channels is %d, the filter's shape is [%s], "
          "the groups is %d.",
          filter_dims[0],
          filter_dims,
          groups));

  if (ctx->IsRuntime()) {
    PADDLE_ENFORCE_GT(
        filter_dims[0],
        0,
        platform::errors::InvalidArgument(
            "the size of filter at axis 0 should be greater than 0"));
  }

  framework::DDim in_data_dims;
  if (channel_last) {
    in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
  } else {
    in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
  }

  framework::DDim filter_data_dims =
      phi::slice_ddim(filter_dims, 2, filter_dims.size());

  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  std::vector<int64_t> output_shape({in_dims[0]});
  if (!channel_last) {
    output_shape.push_back(filter_dims[0]);
  }
  for (int i = 0; i < in_data_dims.size(); ++i) {
    if ((!ctx->IsRuntime()) &&
        (in_data_dims[i] <= 0 || filter_dims[i + 2] <= 0)) {
      output_shape.push_back(-1);
    } else {
      output_shape.push_back(ConvOutputSize(in_data_dims[i],
                                            filter_data_dims[i],
                                            dilations[i],
                                            paddings[2 * i],
                                            paddings[2 * i + 1],
                                            strides[i]));
    }
  }
  if (channel_last) {
    output_shape.push_back(filter_dims[0]);
  }

  return output_shape;
}

phi::KernelKey ConvOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  return GetConvExpectedKernelType(ctx, this);
}

phi::KernelKey ConvOp::GetKernelTypeForVar(
    const std::string& var_name,
    const phi::DenseTensor& tensor,
    const phi::KernelKey& expected_kernel_type) const {
#ifdef PADDLE_WITH_MKLDNN
  // Only input require reshaping, weights and
  // bias are having shape in NCHW order
  if ((var_name == "Input") &&
      (expected_kernel_type.layout() == phi::DataLayout::ONEDNN) &&
      (tensor.layout() != phi::DataLayout::ONEDNN)) {
    auto attrs = Attrs();
    auto ar = paddle::framework::AttrReader(attrs);
    const std::string data_format = ar.Get<std::string>("data_format");
    auto dl = phi::StringToDataLayout(data_format);
    // Some models may have intentionally set "AnyLayout" for conv
    // op. Treat this as NCHW (default data_format value)
    if (dl != phi::DataLayout::kAnyLayout) {
      return phi::KernelKey(tensor.place(), dl, expected_kernel_type.dtype());
    }
  }
#endif
  return phi::KernelKey(
      tensor.place(), tensor.layout(), expected_kernel_type.dtype());
}

void Conv2DOpMaker::Make() {
  AddInput("Input", "(Tensor), input 0 of conv2d op.");
  AddInput("Filter", "(Tensor), input 1 of conv2d op.");
  AddOutput("Output", "(Tensor), output 0 of conv2d op.");
  AddAttr<std::vector<int>>("strides",
                            "(std::vector<int>), attribute 0 for conv2d op.")
      .SetDefault({1, 1});
  AddAttr<std::vector<int>>("paddings",
                            "(std::vector<int>), attribute 1 for conv2d op.")
      .SetDefault({0, 0});
  AddAttr<std::string>("padding_algorithm",
                       "(std::string), attribute 2 for conv2d op.")
      .SetDefault("EXPLICIT");
  AddAttr<std::vector<int>>("dilations",
                            "(std::vector<int>), attribute 3 for conv2d op.")
      .SetDefault({1, 1});
  AddAttr<int>("groups", "(int), attribute 4 for conv2d op.").SetDefault(1);
  AddAttr<std::string>("data_format",
                       "(std::string), attribute 5 for conv2d op.")
      .SetDefault("NCHW");
  AddComment(R"DOC(
TODO: Documentation of conv2d op.
)DOC");
  Apply();
}

}  // namespace operators
}  // namespace paddle
