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

#include "paddle/operators/scatter_op.h"
#include "paddle/framework/ddim.h"

namespace paddle {
namespace operators {

class ScatterOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Ref"),
                   "Input(Ref) of ScatterOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Index"),
                   "Input(Index) of ScatterOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Updates"),
                   "Input(Updates) of ScatterOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of ScatterOp should not be null.");

    auto updates_dims = ctx->GetInputDim("Updates");
    auto ref_dims = ctx->GetInputDim("Ref");
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Index").size(), 1,
                      "Update Index should be 1-D.");
    PADDLE_ENFORCE_EQ(ref_dims.size(), updates_dims.size(),
                      "Reference and Updates should have the same shape size");
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Updates")[0],
                      ctx->GetInputDim("Index")[0],
                      "Updates and Index should have same batch-size.");
    framework::DDim data_dim(updates_dims);
    for (int i = 1; i < data_dim.size(); ++i) {
      PADDLE_ENFORCE_EQ(data_dim[i], updates_dims[i]);
    }
    ctx->SetOutputDim("Out", ref_dims);
  }

 protected:
  framework::OpKernelType GetKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("Ref")->type()),
        ctx.device_context());
  }
};

class ScatterGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim(framework::GradVarName("Updates"),
                      ctx->GetInputDim("Updates"));
    ctx->SetOutputDim(framework::GradVarName("Ref"), ctx->GetInputDim("Ref"));
  }

 protected:
  framework::OpKernelType GetKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("Ref")->type()),
        ctx.device_context());
  }
};

class ScatterOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ScatterOpMaker(framework::OpProto* proto,
                 framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Ref", "The source input of scatter op");
    AddInput("Index",
             "The index input of scatter op where Ref will be updated");
    AddInput("Updates", "The updated value of updates op");
    AddOutput("Out", "The output of add op");
    AddComment(R"DOC(
Scatter Operator by selecting from the first axis,

Out = Ref
Out[Index] = Ref[Index] + Updates
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(scatter, ops::ScatterOp, ops::ScatterOpMaker, scatter_grad,
            ops::ScatterGradOp);
REGISTER_OP_CPU_KERNEL(scatter, ops::ScatterOpKernel<float>);
REGISTER_OP_CPU_KERNEL(scatter_grad, ops::ScatterGradientOpKernel<float>);
