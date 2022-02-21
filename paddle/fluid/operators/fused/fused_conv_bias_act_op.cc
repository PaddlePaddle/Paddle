/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fused/fused_conv_bias_act_op.h"
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/conv_op.h"
#include "paddle/phi/api/all.h"

namespace paddle {
namespace operators {

void FusedConvBiasActOp::InferShape(framework::InferShapeContext* ctx) const {
  std::vector<int64_t> output_shape = ComputeOutputShape(ctx);

  OP_INOUT_CHECK(ctx->HasOutput("Output"), "Output", "Output",
                 "FusedConvBiasAct");
  ctx->SetOutputDim("Output", phi::make_ddim(output_shape));
  ctx->ShareLoD("Input", "Output");
}

framework::OpKernelType FusedConvBiasActOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  return framework::OpKernelType(
      OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
      ctx.device_context());
}

framework::OpKernelType FusedConvBiasActOp::GetKernelTypeForVar(
    const std::string& var_name, const Tensor& tensor,
    const framework::OpKernelType& expected_kernel_type) const {
  return framework::OpKernelType(expected_kernel_type.data_type_,
                                 tensor.place(), tensor.layout());
}

std::vector<int64_t> FusedConvBiasActOp::ComputeOutputShape(
    framework::InferShapeContext* ctx) const {
  OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input",
                 "conv2d_bias_act_fusion");
  OP_INOUT_CHECK(ctx->HasInput("Filter"), "Input", "Filter",
                 "conv2d_bias_act_fusion");

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
        dilations[i], 0,
        platform::errors::InvalidArgument(
            "The dilation of Op(Conv) should be larget than 0, but received "
            "dilation is %d.",
            dilations[i]));
  }
  const std::string data_format = "NCHW";
  bool channel_last = false;

  PADDLE_ENFORCE_EQ(
      in_dims.size() == 4, true,
      platform::errors::InvalidArgument(
          "The input of Op(Conv) should be a 4-D Tensor. But "
          "received: input's dimension is %u, input's shape is [%s].",
          in_dims.size(), in_dims));

  PADDLE_ENFORCE_EQ(
      in_dims.size(), filter_dims.size(),
      platform::errors::InvalidArgument(
          "The input's dimension and filter's dimension of "
          "Op(Conv) should be equal. But received: the input's shape is [%s], "
          "the input's dimension is %d; the filter's shape is [%s],  "
          "the filter's dimension is %d.",
          in_dims, in_dims.size(), filter_dims, filter_dims.size()));

  int stride_size = strides.size();
  for (int i = 0; i < stride_size; ++i) {
    PADDLE_ENFORCE_GT(
        strides[i], 0,
        platform::errors::InvalidArgument(
            "The stride of Op(Conv) should be larget than 0, but received "
            "stride is %d.",
            strides[i]));
  }

  int in_sub_stride_size = in_dims.size() - stride_size;
  PADDLE_ENFORCE_EQ(
      in_dims.size(), strides.size() + 2U,
      platform::errors::InvalidArgument(
          "The difference of input's dimension and Attr(strides)'s "
          "length must be euqal to 2 for Op(Conv). "
          "But received: input's dimension is %d, input's shape is [%s]; "
          "Attr(stride)'s length is %d, Attr(stride) is [%s]; "
          "difference of input's dimention and Attr(strides)'s length = %u.",
          in_dims.size(), in_dims, strides.size(), phi::make_ddim(strides),
          in_sub_stride_size));

  const auto input_channels = in_dims[1];

  PADDLE_ENFORCE_EQ(
      input_channels, filter_dims[1] * groups,
      platform::errors::InvalidArgument(
          "The number of input's channels should be equal to filter's channels "
          "* groups for Op(Conv). But received: the input's channels is %d, "
          "the input's shape is [%s]; the filter's channels is %d, the "
          "filter's shape is [%s]; the groups is %d, the data_format is %s. "
          "The error may come from wrong data_format setting.",
          input_channels, in_dims, filter_dims[1], filter_dims, groups,
          data_format));
  PADDLE_ENFORCE_EQ(
      filter_dims[0] % groups, 0,
      platform::errors::InvalidArgument(
          "The number of output's channels (filter's first dimension) of "
          "Op(Conv) should be divided by groups. But received: "
          "the output channels is %d, the filter's shape is [%s], "
          "the groups is %d.",
          filter_dims[0], filter_dims, groups));

  if (ctx->IsRuntime()) {
    PADDLE_ENFORCE_GT(
        filter_dims[0], 0,
        platform::errors::InvalidArgument(
            "the size of filter at axis 0 should be greater than 0"));
  }

  phi::DDim in_data_dims;
  in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());

  phi::DDim filter_data_dims =
      phi::slice_ddim(filter_dims, 2, filter_dims.size());

  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                           in_data_dims, strides, ksize);

  std::vector<int64_t> output_shape({in_dims[0]});
  if (!channel_last) {
    output_shape.push_back(filter_dims[0]);
  }
  for (int i = 0; i < in_data_dims.size(); ++i) {
    if ((!ctx->IsRuntime()) &&
        (in_data_dims[i] <= 0 || filter_dims[i + 2] <= 0)) {
      output_shape.push_back(-1);
    } else {
      output_shape.push_back(
          ConvOutputSize(in_data_dims[i], filter_data_dims[i], dilations[i],
                         paddings[2 * i], paddings[2 * i + 1], strides[i]));
    }
  }
  if (channel_last) {
    output_shape.push_back(filter_dims[0]);
  }

  return output_shape;
}

void FusedConvBiasActOpMaker::Make() {
  AddInput("Input", "(Tensor) NCHW layout.");
  AddInput("Filter", "(vector<Tensor>) 4 aggregated filters");
  AddInput("Bias", "(vector<Tensor>) it's length is equal to Filter");
  AddOutput("Output",
            "(Tensor) The output tensor of convolution operator. "
            "The format of output tensor is also NCHW.");
  AddAttr<std::string>("activation",
                       "The activation type can be 'linear', 'relu'")
      .SetDefault("relu");
  AddAttr<std::vector<int>>("strides",
                            "(vector<int> default:{1, 1}), the "
                            "strides(h_stride, w_stride) of "
                            "convolution operator.")
      .SetDefault({1, 1});
  AddAttr<std::vector<int>>("paddings",
                            "(vector<int> default:{0, 0}), the "
                            "paddings(pad_height_top, pad_height_bottom, "
                            "pad_width_left, pad_wifth_right)  of "
                            "convolution operator.")
      .SetDefault({0, 0});
  AddAttr<std::vector<int>>("dilations",
                            "(vector<int> default:{1, 1}), the "
                            "dilations(h_dilation, w_dilation) of "
                            "convolution operator.")
      .SetDefault({1, 1});
  AddAttr<int>(
      "groups",
      "(int default:1), the groups number of the convolution operator. "
      "According to grouped convolution in Alex Krizhevsky's Deep CNN paper: "
      "when group=2, the first half of the filters is only connected to the "
      "first half of the input channels, while the second half of the filters "
      "is only connected to the second half of the input channels.")
      .SetDefault(1);
  AddAttr<std::string>(
      "padding_algorithm",
      "(string, default \"EXPLICIT\") An optional string from: \"EXPLICIT\","
      "\"SAME\",\"VALID\". Set to \"EXPLICIT\" for explicit padding. "
      "Set to \"SAME\" or \"VALID\" for algorithm of padding. ")
      .SetDefault("EXPLICIT");
  AddAttr<std::string>(
      "data_format",
      "(string, default NCHW) Only used in "
      "An optional string from: \"NHWC\", \"NCHW\". "
      "Defaults to \"NHWC\". Specify the data format of the output data, "
      "the input will be transformed automatically. ")
      .SetDefault("NCHW");

  AddComment(R"DOC(
)DOC");
}

void FusedConvBiasActGradOp::InferShape(
    framework::InferShapeContext* ctx) const {
  auto in_dims = ctx->GetInputDim("Input");
  auto filter_dims = ctx->GetInputDim("Filter");
  if (ctx->HasOutput(framework::GradVarName("Input"))) {
    ctx->SetOutputDim(framework::GradVarName("Input"), in_dims);
  }
  if (ctx->HasOutput(framework::GradVarName("Filter"))) {
    ctx->SetOutputDim(framework::GradVarName("Filter"), filter_dims);
  }
}

framework::OpKernelType FusedConvBiasActGradOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  return framework::OpKernelType(
      OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
      ctx.device_context());
}

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fused_conv2d_bias_act, ops::FusedConvBiasActOp,
                  ops::FusedConvBiasActOpMaker,
                  ops::FusedConvBiasActOpInferVarType,
                  ops::FusedConvBiasActGradOpMaker<paddle::framework::OpDesc>,
                  ops::FusedConvBiasActGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(fused_conv2d_bias_act_grad, ops::FusedConvBiasActGradOp);
