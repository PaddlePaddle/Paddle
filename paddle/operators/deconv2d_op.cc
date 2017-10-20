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

#include "paddle/operators/deconv2d_op.h"
#include "paddle/operators/conv2d_op.h"

namespace paddle {
namespace operators {

void Deconv2DOp::InferShape(framework::InferShapeContext* ctx) const {
  PADDLE_ENFORCE(ctx->HasInput("Input"),
                 "Input(Input) of Deconv2DOp should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("Filter"),
                 "Input(Filter) of Deconv2DOp should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("Output"),
                 "Output(Output) of Deconv2DOp should not be null.");

  auto in_dims = ctx->GetInputDim("Input");
  auto filter_dims = ctx->GetInputDim("Filter");
  std::vector<int> strides = ctx->Attrs().Get<std::vector<int>>("strides");
  std::vector<int> paddings = ctx->Attrs().Get<std::vector<int>>("paddings");

  for (size_t i = 0; i < paddings.size(); ++i) {
    PADDLE_ENFORCE_EQ(paddings[i], 0, "No Padding allowed in deconv op.");
  }

  PADDLE_ENFORCE_EQ(in_dims.size(), 4,
                    "Deconv2DOp input should be 4-D tensor.");
  PADDLE_ENFORCE_EQ(filter_dims.size(), 4,
                    "Deconv2DOp filter should be 4-D tensor.");
  PADDLE_ENFORCE_EQ(in_dims[1], filter_dims[0],
                    "input and kernel input dimension should be equal.");

  auto output_height = (in_dims[2] - 1) * strides[0] + filter_dims[2];
  auto output_width = (in_dims[3] - 1) * strides[1] + filter_dims[3];
  ctx->SetOutputDim("Output",
                    {in_dims[0], filter_dims[1], output_height, output_width});
}

Deconv2DOpMaker::Deconv2DOpMaker(framework::OpProto* proto,
                                 framework::OpAttrChecker* op_checker)
    : OpProtoAndCheckerMaker(proto, op_checker) {
  AddInput(
      "Input",
      "The input tensor of deconvolution operator. "
      "The format of input tensor is NCHW. Where N is batch size, C is the "
      "number of input channels, H and W is the height and width of image.");
  AddInput("Filter",
           "The filter tensor of deconvolution operator."
           "The format of the filter tensor is MCHW, where C is the number of "
           "output image channels, M is the number of input image channels, "
           "H and W is height and width of filter. "
           "We enforce groups number == 1 and padding == 0 in "
           "deconvolution Scenario.");
  AddOutput("Output",
            "The output tensor of deconvolution operator."
            "The format of output tensor is also NCHW.");
  AddAttr<std::vector<int>>("strides", "strides of deconvolution operator.")
      .SetDefault({1, 1});
  AddAttr<std::vector<int>>("paddings", "paddings of deconvolution operator.")
      .SetDefault({0, 0});
  AddComment(R"DOC(
The deconvolution operation calculates the output based on the input, filter
and strides, paddings, groups parameters. The size of each dimension of the
parameters is checked in the infer-shape.
)DOC");
}

void Deconv2DOpGrad::InferShape(framework::InferShapeContext* ctx) const {
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
REGISTER_OP(deconv2d, ops::Deconv2DOp, ops::Deconv2DOpMaker, deconv2d_grad,
            ops::Deconv2DOpGrad);

REGISTER_OP_CPU_KERNEL(
    deconv2d, ops::GemmDeconv2DKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    deconv2d_grad,
    ops::GemmDeconvGrad2DKernel<paddle::platform::CPUPlace, float>);
