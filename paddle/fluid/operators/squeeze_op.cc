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

#include "paddle/fluid/operators/squeeze_op.h"

namespace paddle {
namespace operators {

class SqueezeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of SqueezeOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of SqueezeOp should not be null.");
    auto x_dims = ctx->GetInputDim("X");
    std::vector<int> axes = ctx->Attrs().Get<std::vector<int>>("axes");
    std::vector<int64_t> output_shape;
    if (axes.size() > 0) {
      for (int i = 0; i < x_dims.size(); ++i) {
        if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
          output_shape.push_back(x_dims[i]);
        }
      }
    } else {
      for (int i = 0; i < x_dims.size(); ++i) {
        if (x_dims[i] != 1) {
          output_shape.push_back(x_dims[i]);
        }
      }
    }

    ctx->SetOutputDim("Out", framework::make_ddim(output_shape));
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class SqueezeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SqueezeOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(Tensor)The input of squeeze op");
    AddOutput("Out", "(Tensor)The output of squeeze op");
    AddAttr<std::vector<int>>("axes",
                              "(vector<int>) "
                              "List of positive integers,"
                              "indicate the dimensions to squeeze.")
        .SetDefault({});
    AddComment(R"DOC(
Squeeze Operator.

Remove single-dimensional entries from the shape of a tensor. 
Takes a parameter axes with a list of axes to squeeze. 

)DOC");
  }
};

class SqueezeOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                   "Output(X@GRAD) should not be null.");
    auto x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(squeeze, ops::SqueezeOp, ops::SqueezeOpMaker, squeeze_grad,
            ops::SqueezeOpGrad);
REGISTER_OP_CPU_KERNEL(
    squeeze, ops::SqueezeKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SqueezeKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    squeeze_grad,
    ops::SqueezeGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SqueezeGradKernel<paddle::platform::CPUDeviceContext, double>);
