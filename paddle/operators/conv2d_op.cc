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

#include "paddle/operators/gemm_conv2d_op.h"

namespace paddle {
namespace operators {

int outputSize(int input_size, int filter_size, int padding, int stride) {
  int output_size = (input_size - filter_size + 2 * padding) / stride + 1;
  return output_size;
}

class Conv2DOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Input"),
                   "Input(Input) of Conv2DOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Filter"),
                   "Input(Filter) of Conv2DOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Output"),
                   "Output(Output) of Conv2DOp should not be null.");

    auto in_dims = ctx->GetInputDim("Input");
    auto filter_dims = ctx->GetInputDim("Filter");
    std::vector<int> strides = ctx->Attrs().Get<std::vector<int>>("strides");
    std::vector<int> paddings = ctx->Attrs().Get<std::vector<int>>("paddings");
    int groups = ctx->Attrs().Get<int>("groups");
    int input_channels = in_dims[1];
    int output_channels = filter_dims[0];

    PADDLE_ENFORCE_EQ(in_dims.size(), 4, "Conv2DOp input should be 4-D.");
    PADDLE_ENFORCE_EQ(filter_dims.size(), 4, "Conv2DOp filter should be 4-D.");
    PADDLE_ENFORCE_EQ(input_channels, filter_dims[1] * groups,
                      "The number of input channels should be equal to filter "
                      "channels * groups.");
    PADDLE_ENFORCE_EQ(
        output_channels % groups, 0,
        "The number of output channels should be divided by groups.");

    auto output_height =
        outputSize(in_dims[2], filter_dims[2], paddings[0], strides[0]);
    auto output_width =
        outputSize(in_dims[3], filter_dims[3], paddings[1], strides[1]);
    ctx->SetOutputDim(
        "Output", {in_dims[0], filter_dims[0], output_height, output_width});
  }
};

class Conv2DOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  Conv2DOpMaker(framework::OpProto* proto, framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(
        "Input",
        "The input tensor of convolution operator. "
        "The format of input tensor is NCHW. Where N is batch size, C is the "
        "number of channels, H and W is the height and width of image.");
    AddInput(
        "Filter",
        "The filter tensor of convolution operator."
        "The format of the filter tensor is MCHW, where M is the number of "
        "output image channels, C is the number of input image channels, "
        "H and W is height and width of filter. "
        "If the groups attribute is greater than 1, C equal the number of "
        "input image channels divided by the groups.");
    AddOutput("Output",
              "The output tensor of convolution operator."
              "The format of output tensor is also NCHW.");
    AddAttr<std::vector<int>>("strides", "strides of convolution operator.")
        .SetDefault({1, 1});
    AddAttr<std::vector<int>>("paddings", "paddings of convolution operator.")
        .SetDefault({0, 0});
    AddAttr<int>(
        "groups",
        "group size of convolution operator. "
        "Refer to grouped convolution in Alex Krizhevsky's paper: "
        "when group=2, the first half of the filters are only connected to the "
        "first half of the input channels, and the second half only connected "
        "to the second half.")
        .SetDefault(1);
    AddComment(R"DOC(
The convolution operation calculates the output based on the input, filter
and strides, paddings, groups parameters. The size of each dimension of the
parameters is checked in the infer-shape.
)DOC");
  }
};

class Conv2DOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    auto in_dims = ctx->GetInputDim("Input");
    auto filter_dims = ctx->GetInputDim("Filter");
    if (ctx->HasOutput(framework::GradVarName("Input"))) {
      ctx->SetOutputDim(framework::GradVarName("Input"), in_dims);
    }
    if (ctx->HasOutput(framework::GradVarName("Filter"))) {
      ctx->SetOutputDim(framework::GradVarName("Filter"), filter_dims);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(conv2d, ops::Conv2DOp, ops::Conv2DOpMaker, conv2d_grad,
            ops::Conv2DOpGrad);

REGISTER_OP_CPU_KERNEL(
    conv2d, ops::GemmConv2DKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    conv2d_grad, ops::GemmConvGrad2DKernel<paddle::platform::CPUPlace, float>);
