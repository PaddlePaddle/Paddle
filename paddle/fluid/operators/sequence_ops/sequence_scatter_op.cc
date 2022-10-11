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

#include "paddle/fluid/operators/sequence_ops/sequence_scatter_op.h"

#include <memory>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;
using LoDTensor = framework::LoDTensor;

class SequenceScatterOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) The source input of sequence scatter op");
    AddInput("Ids",
             "(LoDTensor) The index input of sequence scatter op where X"
             " will be  updated, must be a LoDTensor");
    AddInput("Updates",
             "(LoDTensor) The values to scatter to the input tensor "
             "X, must be a LoDTensor with the same LoD information as Ids");
    AddOutput("Out",
              "(Tensor) The output tensor of sequence scatter op, which "
              "has the same dims as X");
    AddComment(R"DOC(
Sequence Scatter Operator.

This operator scatters the Updates tensor to the input X. It uses the LoD
information of Ids to select the rows to update, and use the values in Ids as
the columns to update in each row of X.

Following are cases to better explain how this works:

Example 1:
Given an all-ones Tensor input(X)
    X.data = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
              [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
              [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    X.dims = [3, 6]
a LoDTensor input(Ids)
    Ids.data = [[0], [1], [2], [5], [4], [3], [2], [1], [3], [2], [5], [4]]
    Ids.lod =  [[0,        3,                       8,                 12]]
and a Tensor input(Updates)
    Updates.data = [[0.3], [0.3], [0.4], [0.1], [0.2], [0.3], [0.4], [0.0], [0.2], [0.3], [0.1], [0.4]]
    Updates.lod =  [[  0,            3,                                 8,                         12]]
then we get an output Tensor
    Out.data = [[1.3, 1.3, 1.4, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.4, 1.3, 1.2, 1.1],
                [1.0, 1.0, 1.3, 1.2, 1.4, 1.1]]
    Out.dims = X.dims = [3, 6]
)DOC");
  }
};

class SequenceScatterOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    // Enforce has inputs and outputs
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "SequenceScatter");
    OP_INOUT_CHECK(ctx->HasInput("Ids"), "Input", "Ids", "SequenceScatter");
    OP_INOUT_CHECK(
        ctx->HasInput("Updates"), "Input", "Updates", "SequenceScatter");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "SequenceScatter");

    // Set output dim the same as input
    auto ref_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", ref_dims);

    // Enforce the Updates and Ids are the same shape
    auto updates_dim = ctx->GetInputDim("Updates");
    auto ids_dim = ctx->GetInputDim("Ids");
    PADDLE_ENFORCE_EQ(
        updates_dim[0],
        ids_dim[0],
        platform::errors::InvalidArgument(
            "The shape of SequenceScatter operator's input Updates and Ids do "
            "not match, receive Updates's shape is [%s], Ids's shape is [%s].",
            updates_dim,
            ids_dim));

    // Enforce LoD of ids and updates be the same
    if (ctx->IsRuntime()) {
      framework::Variable* ids_var =
          PADDLE_GET(framework::Variable*, ctx->GetInputVarPtrs("Ids")[0]);
      framework::Variable* updates_var =
          PADDLE_GET(framework::Variable*, ctx->GetInputVarPtrs("Updates")[0]);

      auto& ids_lod = ids_var->Get<LoDTensor>().lod();
      auto& updates_lod = updates_var->Get<LoDTensor>().lod();
      PADDLE_ENFORCE_EQ(
          ids_lod.size(),
          1,
          platform::errors::InvalidArgument(
              "The SequenceScatter operator’s Input Ids holds wrong LoD "
              "information. Currently SequenceScatter operator can only deal "
              "with one level LoD for input Ids, but received LoD level is %d.",
              ids_lod.size()));
      PADDLE_ENFORCE_EQ(
          updates_lod.size(),
          1,
          platform::errors::InvalidArgument(
              "The SequenceScatter operator’s Input Updates holds wrong LoD "
              "information. Currently SequenceScatter operator can only deal "
              "with one level LoD for input Updates, but received LoD level is "
              "%d.",
              ids_lod.size()));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        platform::CPUPlace());
  }
};

class SequenceScatterGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim(framework::GradVarName("Updates"),
                      ctx->GetInputDim("Updates"));
    ctx->SetOutputDim(framework::GradVarName("X"),
                      ctx->GetInputDim(framework::GradVarName("Out")));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   platform::CPUPlace());
  }
};

template <typename T>
class SequenceScatterGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("sequence_scatter_grad");
    op->SetInput("Ids", this->Input("Ids"));
    op->SetInput("Updates", this->Input("Updates"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Updates"),
                  this->InputGrad("Updates"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(SequenceScatterGradNoNeedBufferVarsInferer,
                                    "Updates");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(sequence_scatter,
                  ops::SequenceScatterOp,
                  ops::SequenceScatterOpMaker,
                  ops::SequenceScatterGradMaker<paddle::framework::OpDesc>,
                  ops::SequenceScatterGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(sequence_scatter_grad,
                  ops::SequenceScatterGradOp,
                  ops::SequenceScatterGradNoNeedBufferVarsInferer);
REGISTER_OP_CPU_KERNEL(sequence_scatter,
                       ops::SequenceScatterOpKernel<float>,
                       ops::SequenceScatterOpKernel<double>,
                       ops::SequenceScatterOpKernel<int>,
                       ops::SequenceScatterOpKernel<int64_t>);
REGISTER_OP_CPU_KERNEL(sequence_scatter_grad,
                       ops::SequenceScatterGradientOpKernel<float>,
                       ops::SequenceScatterGradientOpKernel<double>,
                       ops::SequenceScatterGradientOpKernel<int>,
                       ops::SequenceScatterGradientOpKernel<int64_t>);
