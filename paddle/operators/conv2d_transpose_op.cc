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

#include "paddle/operators/conv2d_transpose_op.h"

namespace paddle {
namespace operators {

void Conv2DTransposeOp::InferShape(framework::InferShapeContext* ctx) const {
  PADDLE_ENFORCE(ctx->HasInput("Input"),
                 "Input(Input) of Conv2DTransposeOp should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("Filter"),
                 "Input(Filter) of Conv2DTransposeOp should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("Output"),
                 "Output(Output) of Conv2DTransposeOp should not be null.");

  auto in_dims = ctx->GetInputDim("Input");
  auto filter_dims = ctx->GetInputDim("Filter");
  std::vector<int> strides = ctx->Attrs().Get<std::vector<int>>("strides");
  std::vector<int> paddings = ctx->Attrs().Get<std::vector<int>>("paddings");

  for (size_t i = 0; i < paddings.size(); ++i) {
    PADDLE_ENFORCE_EQ(paddings[i], 0,
                      "No Padding allowed in conv transpose op.");
  }

  PADDLE_ENFORCE_EQ(in_dims.size(), 4,
                    "Conv2DTransposeOp input should be 4-D tensor.");
  PADDLE_ENFORCE_EQ(filter_dims.size(), 4,
                    "Conv2DTransposeOp filter should be 4-D tensor.");
  PADDLE_ENFORCE_EQ(in_dims[1], filter_dims[0],
                    "input and kernel input dimension should be equal.");

  auto output_height = (in_dims[2] - 1) * strides[0] + filter_dims[2];
  auto output_width = (in_dims[3] - 1) * strides[1] + filter_dims[3];
  ctx->SetOutputDim("Output",
                    {in_dims[0], filter_dims[1], output_height, output_width});
}

Conv2DTransposeOpMaker::Conv2DTransposeOpMaker(
    framework::OpProto* proto, framework::OpAttrChecker* op_checker)
    : OpProtoAndCheckerMaker(proto, op_checker) {
  AddInput(
      "Input",
      "(Tensor) The input tensor of convolution transpose operator. "
      "The format of input tensor is NCHW, where N is batch size, C is the "
      "number of input channels, H is the height of the image, and "
      "W is the width of the image.");
  AddInput("Filter",
           "(Tensor) The filter tensor of convolution transpose operator."
           "The format of the filter tensor is CMHW, where C is the number of "
           "output image channels, M is the number of input image channels, "
           "H is the height of the filter, and W is the width of the filter. "
           "We enforce groups number == 1 and padding == 0 in "
           "the convolution transpose Scenario.");
  AddOutput("Output",
            "(Tensor) The output tensor of convolution transpose operator."
            "The format of output tensor is also NCHW.");
  AddAttr<std::vector<int>>("strides",
                            "strides of convolution transpose operator.")
      .SetDefault({1, 1});
  AddAttr<std::vector<int>>("paddings",
                            "paddings of convolution transpose operator.")
      .SetDefault({0, 0});
  AddComment(R"DOC(
Convolution Transpose Operator.

The convolution transpose operation calculates the output based on the input, 
filter, strides, paddings, and groups parameters. The size of each dimension 
of the parameters is checked in the infer-shape method.

)DOC");
}

void Conv2DTransposeOpGrad::InferShape(
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

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(conv2d_transpose, ops::Conv2DTransposeOp,
            ops::Conv2DTransposeOpMaker, conv2d_transpose_grad,
            ops::Conv2DTransposeOpGrad);

REGISTER_OP_CPU_KERNEL(
    conv2d_transpose,
    ops::GemmConv2DTransposeKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    conv2d_transpose_grad,
    ops::GemmConv2DTransposeGradKernel<paddle::platform::CPUPlace, float>);
