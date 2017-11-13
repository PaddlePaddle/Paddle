/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/operators/conv_op.h"

namespace paddle {
namespace operators {

void ConvOp::InferShape(framework::InferShapeContext* ctx) const {
  PADDLE_ENFORCE(ctx->HasInput("Input"),
                 "Input(Input) of ConvOp should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("Filter"),
                 "Input(Filter) of ConvOp should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("Output"),
                 "Output(Output) of ConvOp should not be null.");

  auto in_dims = ctx->GetInputDim("Input");
  auto filter_dims = ctx->GetInputDim("Filter");
  std::vector<int> strides = ctx->Attrs().Get<std::vector<int>>("strides");
  std::vector<int> paddings = ctx->Attrs().Get<std::vector<int>>("paddings");
  int groups = ctx->Attrs().Get<int>("groups");
  int input_channels = in_dims[1];
  int output_channels = filter_dims[0];

  PADDLE_ENFORCE(in_dims.size() == 4 || in_dims.size() == 5,
                 "Conv intput should be 4-D or 5-D tensor.");
  PADDLE_ENFORCE_EQ(
      in_dims.size(), filter_dims.size(),
      "Conv input dimension and filter dimension should be the same.");
  PADDLE_ENFORCE(
      in_dims.size() - strides.size() == 2U,
      "Conv input dimension and strides dimension should be consistent.");
  PADDLE_ENFORCE_EQ(
      paddings.size(), strides.size(),
      "Conv paddings dimension and Conv strides dimension should be the same.");
  PADDLE_ENFORCE_EQ(input_channels, filter_dims[1] * groups,
                    "The number of input channels should be equal to filter "
                    "channels * groups.");
  PADDLE_ENFORCE_EQ(
      output_channels % groups, 0,
      "The number of output channels should be divided by groups.");

  std::vector<int64_t> output_shape({in_dims[0], filter_dims[0]});
  for (size_t i = 0; i < paddings.size(); ++i) {
    output_shape.push_back(OutputSize(in_dims[i + 2], filter_dims[i + 2],
                                      paddings[i], strides[i]));
  }
  ctx->SetOutputDim("Output", framework::make_ddim(output_shape));
}

Conv2DOpMaker::Conv2DOpMaker(framework::OpProto* proto,
                             framework::OpAttrChecker* op_checker)
    : OpProtoAndCheckerMaker(proto, op_checker) {
  AddInput(
      "Input",
      "(Tensor) The input tensor of convolution operator. "
      "The format of input tensor is NCHW, where N is batch size, C is the "
      "number of channels, H is the height of the feature, "
      "and W is the width of the feature.");
  AddInput("Filter",
           "(Tensor) The filter tensor of convolution operator. "
           "The format of the filter tensor is MCHW, where M is the number of "
           "output image channels, C is the number of input image channels, "
           "H is the height of the filter, and W is the width of the filter. "
           "If the groups attribute is greater than 1, C equals the number of "
           "input image channels divided by the groups.");
  AddOutput("Output",
            "(Tensor) The output tensor of convolution operator. "
            "The format of output tensor is also NCHW.");
  AddAttr<std::vector<int>>("strides", "strides of convolution operator.")
      .SetDefault({1, 1});
  AddAttr<std::vector<int>>("paddings", "paddings of convolution operator.")
      .SetDefault({0, 0});
  AddAttr<int>(
      "groups",
      "(int default:1), the group size of convolution operator. "
      "According to grouped convolution in Alex Krizhevsky's Deep CNN paper: "
      "when group=2, the first half of the filters is only connected to the "
      "first half of the input channels, while the second half of the filters "
      "is only connected to the second half of the input channels.")
      .SetDefault(1);
  AddComment(R"DOC(
Convolution Operator.

The convolution operation calculates the output based on the input, filter
and strides, paddings, groups parameters. The size of each dimension of the
parameters is checked in the infer-shape.
Input(Input, Filter) and output(Output) are in NCHW format. Where N is batch
size, C is the number of channels, H is the height of the feature, and W is
the width of the feature. Parameters(ksize, strides, paddings) are two elements.
These two elements represent height and width, respectively.
The input(X) size and output(Out) size may be different.

Example:
  Input:
       Input shape: (N, C_in, H_in, W_in)
       Filter shape: (C_out, C_in, H_f, W_f)
  Output:
       Output shape: (N, C_out, H_out, W_out)
  where
       H_out = (H_in - filter_size[0] + 2 * paddings[0]) / strides[0] + 1;
       W_out = (W_in - filter_size[1] + 2 * paddings[1]) / strides[1] + 1;
)DOC");
}

Conv3DOpMaker::Conv3DOpMaker(framework::OpProto* proto,
                             framework::OpAttrChecker* op_checker)
    : OpProtoAndCheckerMaker(proto, op_checker) {
  AddInput(
      "Input",
      "(Tensor) The input tensor of convolution operator. "
      "The format of input tensor is NCDHW. Where N is batch size, C is the "
      "number of channels, D is the depth of the feature, H is the height of "
      "the feature, "
      "and W is the width of the feature.");
  AddInput("Filter",
           "(Tensor) The filter tensor of convolution operator. "
           "The format of the filter tensor is MCDHW, where M is the number of "
           "output image channels, C is the number of input image channels, "
           "D is the depth of the filter, H is the height of the filter, and W "
           "is the width of the filter."
           "If the groups attribute is greater than 1, C equals the number of "
           "input image channels divided by the groups.");
  AddOutput("Output",
            "(Tensor) The output tensor of convolution operator."
            "The format of output tensor is also NCDHW.");
  AddAttr<std::vector<int>>(
      "strides",
      "(vector, default:{0, 0, 0}), the strides of convolution operator.")
      .SetDefault({1, 1, 1});
  AddAttr<std::vector<int>>(
      "paddings",
      "(vector, default:{0, 0, 0}), the paddings of convolution operator.")
      .SetDefault({0, 0, 0});
  AddAttr<int>(
      "groups",
      "(int default:1), the group size of convolution operator. "
      "According to grouped convolution in Alex Krizhevsky's Deep CNN paper: "
      "when group=2, the first half of the filters is only connected to the "
      "first half of the input channels, while the second half of the filters "
      "is only connected to the second half of the input channels.")
      .SetDefault(1);

  AddComment(R"DOC(
Convolution3D Operator.

The convolution operation calculates the output based on the input, filter
and strides, paddings, groups parameters. The size of each dimension of the
parameters is checked in the infer-shape.
Input(Input, Filter) and output(Output) are in NCDHW format. Where N is batch
size, C is the number of channels,D is the depth of the feature, H is the height of
the feature, and W is the width of the feature. Parameters(ksize, strides, paddings)
are three elements. These three elements represent depth, height and width, respectively.
The input(X) size and output(Out) size may be different.

Example:
  Input:
       Input shape: (N, C_in, D_in, H_in, W_in)
       Filter shape: (C_out, C_in, D_f, H_f, W_f)
  Output:
       Output shape: (N, C_out, D_out, H_out, W_out)
  where
       D_out = (D_in - filter_size[0] + 2 * paddings[0]) / strides[0] + 1;
       H_out = (H_in - filter_size[1] + 2 * paddings[1]) / strides[1] + 1;
       W_out = (W_in - filter_size[2] + 2 * paddings[2]) / strides[2] + 1;
)DOC");
}

void ConvOpGrad::InferShape(framework::InferShapeContext* ctx) const {
  auto in_dims = ctx->GetInputDim("Input");
  auto filter_dims = ctx->GetInputDim("Filter");
  if (ctx->HasOutput(framework::GradVarName("Input"))) {
    ctx->SetOutputDim(framework::GradVarName("Input"), in_dims);
  }
  if (ctx->HasOutput(framework::GradVarName("Filter"))) {
    ctx->SetOutputDim(framework::GradVarName("Filter"), filter_dims);
  }
}

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(conv2d, ops::ConvOp, ops::Conv2DOpMaker, conv2d_grad,
            ops::ConvOpGrad);
namespace ops = paddle::operators;
REGISTER_OP(conv3d, ops::ConvOp, ops::Conv3DOpMaker, conv3d_grad,
            ops::ConvOpGrad);

REGISTER_OP_CPU_KERNEL(conv2d,
                       ops::GemmConvKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    conv2d_grad, ops::GemmConvGradKernel<paddle::platform::CPUPlace, float>);

REGISTER_OP_CPU_KERNEL(conv3d,
                       ops::GemmConvKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    conv3d_grad, ops::GemmConvGradKernel<paddle::platform::CPUPlace, float>);
