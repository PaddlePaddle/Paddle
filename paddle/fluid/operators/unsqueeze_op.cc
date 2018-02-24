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

#include "paddle/fluid/operators/unsqueeze_op.h"

namespace paddle {
namespace operators {

class UnsqueezeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of UnsqueezeOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of UnsqueezeOp should not be null.");
    auto x_dims = ctx->GetInputDim("X");
    std::vector<int> axes = ctx->Attrs().Get<std::vector<int>>("axes");
    PADDLE_ENFORCE_GT(axes.size(), 0,
                      "the size of axes should be greater than 0.");
    std::vector<int64_t> output_shape;
    size_t x_index = 0;
    for (int i = 0; i < int(x_dims.size() + axes.size()); ++i) {
      if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
        output_shape.push_back(x_dims[x_index]);
        ++x_index;
      } else {
        output_shape.push_back(1);
      }
    }

    ctx->SetOutputDim("Out", framework::make_ddim(output_shape));
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class UnsqueezeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  UnsqueezeOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(Tensor)The input of unsqueeze op");
    AddOutput("Out", "(Tensor)The output of unsqueeze op");
    AddAttr<std::vector<int>>("axes",
                              "(vector<int>) "
                              "List of positive integers,"
                              "indicate the dimensions to be inserted.");
    AddComment(R"DOC(
Unsqueeze Operator.

Insert single-dimensional entries to the shape of a tensor.
Takes one required argument axes, a list of dimensions that will be inserted.

)DOC");
  }
};

class UnsqueezeOpGrad : public framework::OperatorWithKernel {
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
REGISTER_OP(unsqueeze, ops::UnsqueezeOp, ops::UnsqueezeOpMaker, unsqueeze_grad,
            ops::UnsqueezeOpGrad);
REGISTER_OP_CPU_KERNEL(
    unsqueeze, ops::UnsqueezeKernel<paddle::platform::CPUDeviceContext, float>,
    ops::UnsqueezeKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    unsqueeze_grad,
    ops::UnsqueezeGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::UnsqueezeGradKernel<paddle::platform::CPUDeviceContext, double>);
