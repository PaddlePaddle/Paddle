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

#include "paddle/operators/multiplex_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class MultiplexOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Ids"), "Input(Ids) shouldn't be null.");
    PADDLE_ENFORCE(!ctx->Inputs("X").empty(),
                   "MultiInput(X) shouldn't be empty.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Output(Out) shouldn't be null.");
    auto ids_dim = ctx->GetInputDim("Ids");
    PADDLE_ENFORCE(
        ids_dim.size() == 2 && ids_dim[1] == 1,
        "The index tensor must be a vector with size batchSize x 1.");

    auto ins_dims = ctx->GetInputsDim("X");
    auto num_ins = ins_dims.size();
    PADDLE_ENFORCE(num_ins > 1,
                   "multiplex operator should have more than "
                   "one candidate input tensors.");

    auto in_dim = ins_dims[0];
    PADDLE_ENFORCE(in_dim.size() >= 2,
                   "The rank of candidate tensors must be not less than 2.");
    for (size_t i = 1; i < num_ins; i++) {
      auto dim = ins_dims[i];
      PADDLE_ENFORCE(in_dim == dim,
                     "All the candidate tensors must have the same size.");
    }
    ctx->SetOutputDim("Out", in_dim);
  }

 protected:
  framework::OpKernelType GetKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.MultiInput<Tensor>("X")[0]->type()),
        ctx.device_context());
  }
};

class MultiplexOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  MultiplexOpMaker(framework::OpProto* proto,
                   framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Ids", "The index tensor of multiplex operator.");
    AddInput("X", "The candidate tensors of multiplex operator.")
        .AsDuplicable();
    AddOutput("Out", "The output tensor of multiplex operator.");
    AddComment(R"DOC(
Multiplex Operator.

Multiplex multiple tensors according to the index provided by the index tensor.

Ids: the index tensor.
X[0 : N - 1]: the candidate tensors for output (N >= 2).
For each index i from 0 to batchSize - 1, the output is the i-th row of the
the (Ids[i])-th tensor.

For i-th row of the output tensor:

$$y[i] = x_{k}[i]$$

where `y` is the output tensor, `x_{k}` is the k-th input tensor,
and `k = Ids[i]`.

)DOC");
  }
};

class MultiplexGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(!ctx->Inputs("X").empty(), "Input(X) should not be null.");
    PADDLE_ENFORCE(!ctx->Outputs(framework::GradVarName("X")).empty(),
                   "Output(X@Grad) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null.");
    std::vector<framework::DDim> d_ins;
    auto ins = ctx->GetInputsDim("X");
    // No need to compute gradient for Input(Ids)
    for (size_t i = 0; i < ins.size(); i++) {
      d_ins.push_back(ins[i]);
    }
    ctx->SetOutputsDim(framework::GradVarName("X"), d_ins);
  }

 protected:
  framework::OpKernelType GetKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.MultiInput<Tensor>("X")[0]->type()),
        ctx.device_context());
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OPERATOR(multiplex, ops::MultiplexOp, ops::MultiplexOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<false>);
REGISTER_OPERATOR(multiplex_grad, ops::MultiplexGradOp);
REGISTER_OP_CPU_KERNEL(
    multiplex, ops::MultiplexCPUKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    multiplex_grad,
    ops::MultiplexGradCPUKernel<paddle::platform::CPUPlace, float>);
