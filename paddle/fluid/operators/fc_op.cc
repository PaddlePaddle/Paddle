/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fc_op.h"
#include <vector>

namespace paddle {
namespace operators {

void FCOp::InferShape(framework::InferShapeContext* ctx) const {
  PADDLE_ENFORCE(ctx->HasInput("Input"),
                 "X(Input) of Fully Connected should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("Out"),
                 "Out(Output) of Fully Connected should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("W"),
                 "W(Input) of Fully Connected should not be null.");

  auto in_dims = ctx->GetInputDim("Input");
  auto w_dims = ctx->GetInputDim("W");
  std::vector<int64_t> output_shape({in_dims[0], w_dims[1]});

  PADDLE_ENFORCE(in_dims.size() == 2 || in_dims.size() == 4,
                 "Fully Connected input should be 2-D or 4-D tensor.");

  PADDLE_ENFORCE(w_dims.size() == 2 || w_dims.size() == 4,
                 "Fully Connected input should be 2-D or 4-D tensor.");

  ctx->SetOutputDim("Out", framework::make_ddim(output_shape));
  ctx->ShareLoD("Input", "Out");
}

framework::OpKernelType FCOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  framework::LibraryType library{framework::LibraryType::kMKLDNN};
  framework::DataLayout layout{framework::DataLayout::kAnyLayout};

  return framework::OpKernelType(
      framework::ToDataType(ctx.Input<Tensor>("Input")->type()), ctx.GetPlace(),
      layout, library);
}

void FCOpGrad::InferShape(framework::InferShapeContext* ctx) const {
  auto in_dims = ctx->GetInputDim("Input");
  auto w_dims = ctx->GetInputDim("W");

  if (ctx->HasOutput(framework::GradVarName("Input"))) {
    ctx->SetOutputDim(framework::GradVarName("Input"), in_dims);
  }
  if (ctx->HasOutput(framework::GradVarName("W"))) {
    ctx->SetOutputDim(framework::GradVarName("W"), w_dims);
  }
}

framework::OpKernelType FCOpGrad::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  framework::LibraryType library{framework::LibraryType::kMKLDNN};
  framework::DataLayout layout{framework::DataLayout::kAnyLayout};

  return framework::OpKernelType(
      framework::ToDataType(ctx.Input<Tensor>("Input")->type()), ctx.GetPlace(),
      layout, library);
}

FCOpMaker::FCOpMaker(OpProto* proto, OpAttrChecker* op_checker)
    : OpProtoAndCheckerMaker(proto, op_checker) {
  AddInput("Input", "(Tensor) The input tensor of fully connected operator. ");
  AddInput("W", "(Tensor), The second input tensor of fc op.");
  AddOutput("Out", "(Tensor) The output tensor of fully connected operator. ");
  AddAttr<bool>("use_mkldnn",
                "(bool, default false) Only used in mkldnn kernel")
      .SetDefault(false);
  AddAttr<bool>("bias_attr", "(bool, default false) Only used in mkldnn kernel")
      .SetDefault(false);
  AddComment(R"DOC(
  Fully Connected Operator.

  The fully connected operation calculates the output based on the input, weights and bias attribute.
  The size of each dimension of the parameters checked in the infer-shape.
  The matrix of bias is generated by the mkldnn framework, when the bias_attr is True.
  Additional parametrs are use_mkldnn and bias_attr.
  The input(X) size and output(Out) size may be diffrent.

  The fully connected layer only supports MKLDNN version
)DOC");
}

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(fc, paddle::operators::FCOp, paddle::operators::FCOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(fc_grad, paddle::operators::FCOpGrad);
