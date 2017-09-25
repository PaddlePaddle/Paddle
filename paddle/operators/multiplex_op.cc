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

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Ids"),
                            "Input(Ids) shouldn't be null.");
    PADDLE_ENFORCE(!ctx.MultiInputVar("X").empty(),
                   "MultiInput(X) shouldn't be empty.");
    PADDLE_ENFORCE_NOT_NULL(ctx.OutputVar("Out"),
                            "Output(Out) shouldn't be null.");
    auto ids_dim = ctx.Input<Tensor>("Ids")->dims();
    PADDLE_ENFORCE(
        ids_dim.size() == 2 && ids_dim[1] == 1,
        "The index tensor must be a vector with size batchSize x 1.");

    auto ins = ctx.MultiInput<Tensor>("X");
    auto *out = ctx.Output<Tensor>("Out");
    auto num_ins = ins.size();
    PADDLE_ENFORCE(num_ins > 1,
                   "multiplex operator should have more than "
                   "one candidate input tensors.");

    auto in_dim = ins[0]->dims();
    PADDLE_ENFORCE(in_dim.size() == 2, "Candidate tensors must be matrix.");
    for (size_t i = 1; i < num_ins; i++) {
      auto dim = ins[i]->dims();
      PADDLE_ENFORCE(in_dim == dim,
                     "All the candidate tensors must have the same size.");
    }
    out->Resize(in_dim);
  }
};

class MultiplexOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  MultiplexOpMaker(framework::OpProto *proto,
                   framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Ids", "The index tensor of multiplex operator.");
    AddInput("X", "The candidate tensors of multiplex operator.")
        .AsDuplicable();
    AddOutput("Out", "The output tensor of multiplex operator.");
    AddComment(R"DOC(Multiplex operator

Multiplex multiple tensors according to the index provided by the first
input tensor.

Ids: the index tensor.
X[0 : N - 1]: the candidate tensors for output (N >= 2).
For each index i from 0 to batchSize - 1, the output is the i-th row of the
the (Ids[i])-th tensor.

For i-th row of the output tensor:

y[i][j] = x_{k}[i][j], j = 0,1, ... , (x_{0}.width - 1)

where y is the output tensor. `x_{k}` is the k-th input tensor
and `k = Ids[i]`.
)DOC");
  }
};

class MultiplexGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE(!ctx.MultiInputVar("X").empty(),
                   "Input(X) should not be null.");
    PADDLE_ENFORCE(!ctx.MultiOutputVar(framework::GradVarName("X")).empty(),
                   "Output(X@Grad) should not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar(framework::GradVarName("Out")),
                            "Input(Out@GRAD) shouldn't be null.");
    auto d_ins = ctx.MultiOutput<Tensor>(framework::GradVarName("X"));
    auto ins = ctx.MultiInput<Tensor>("X");
    // No need to compute gradient for Input(Ids)
    for (size_t i = 0; i < ins.size(); i++) {
      if (d_ins[i]) {
        d_ins[i]->Resize(ins[i]->dims());
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OP(multiplex, ops::MultiplexOp, ops::MultiplexOpMaker, multiplex_grad,
            ops::MultiplexGradOp);
REGISTER_OP_CPU_KERNEL(
    multiplex, ops::MultiplexCPUKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    multiplex_grad,
    ops::MultiplexGradCPUKernel<paddle::platform::CPUPlace, float>);
