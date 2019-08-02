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

#include "paddle/fluid/operators/scatter_nd_op.h"
#include <memory>
#include "paddle/fluid/framework/ddim.h"

namespace paddle {
namespace operators {

class ScatterNDOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of ScatterOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Ids"),
                   "Input(Ids) of ScatterOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Updates"),
                   "Input(Updates) of ScatterOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of ScatterOp should not be null.");
    int dim = ctx->Attrs().Get<int>("dim");

    auto updates_dims = ctx->GetInputDim("Updates");
    auto ref_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Ids"), ctx->GetInputDim("Updates"),
                      "Update and Ids must have same dimension.");
    PADDLE_ENFORCE_EQ(ref_dims.size(), updates_dims.size(),
                      "Input and Updates should have the same shape size");
    for (int i = 0; i < ref_dims.size(); i++) {
      if (i != dim)
        PADDLE_ENFORCE_EQ(
            ctx->GetInputDim("Updates")[i], ctx->GetInputDim("X")[i],
            "Updates and Input should have same dimension in %d.", i);
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("Updates")[i],
                        ctx->GetInputDim("Ids")[i],
                        "Updates and Ids should have same dimension in %d.", i);
    }
    ctx->SetOutputDim("Out", ref_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   ctx.device_context());
  }
};

class ScatterNDOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The source input of ScatterND op");
    AddInput("Ids", "The index input of scatter op where X will be updated");
    AddInput("Updates", "The updated value of scatter op");
    AddOutput("Out", "The output of scatter op");
    AddAttr<int>("dim",
                 "(int, defalut: 0) "
                 "the dim to scatter.")
        .SetDefault(0);
    AddComment(R"DOC(
Scatter Operator.

This operator obtains output by updating the input on selected indices on the dim:

$$
Out = X \\
Out[Ids[i][j]][j] = Updates[i][j] dim=0
Out[i][Ids[i][j]] = Updates[i][j] dim=1
$$

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(scatter_nd, ops::ScatterNDOp, ops::ScatterNDOpMaker)
//                  paddle::framework::EmptyGradOpMaker);
//                  ops::ScatterGradDescMaker);
// REGISTER_OPERATOR(scatter_grad, ops::ScatterGradOp,
//                  ops::ScatterGradNoNeedBufferVarsInference);
REGISTER_OP_CPU_KERNEL(scatter_nd, ops::ScatterNDOpKernel<float>,
                       ops::ScatterNDOpKernel<double>,
                       ops::ScatterNDOpKernel<int>);
// REGISTER_OP_CPU_KERNEL(scatter_grad, ops::ScatterGradientOpKernel<float>);
