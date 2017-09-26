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
using LoDTensor = framework::LoDTensor;

class MultiplexOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase &ctx) const override {
    PADDLE_ENFORCE(!ctx.Inputs("X").empty(), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx.HasOutput("Out"), "Output(Out) shouldn't be null.");
    auto ins = ctx.GetInputsDim("X");
    auto num_ins = ins.size();
    PADDLE_ENFORCE(num_ins > 2,
                   "multiplex operator should have more than 2 inputs.");
    PADDLE_ENFORCE_EQ(ins[0].size(), 1,
                      "The first input must be a index vector.");
    auto in_dim = ins[1];
    for (size_t i = 2; i < num_ins; i++) {
      PADDLE_ENFORCE(
          in_dim == ins[i],
          "All the input tensors except the first one must have the same size");
    }
    ctx.SetOutputDim("Out", in_dim);
  }
};

class MultiplexOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  MultiplexOpMaker(framework::OpProto *proto,
                   framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input tensors of multiplex operator.").AsDuplicable();
    AddOutput("Out", "The output tensor of multiplex operator.");
    AddComment(R"DOC(Multiplex operator

Multiplex multiple tensors according to the index provided by the first
input tensor.

ins[0]: the index tensor.
ins[1:N]: the candidate output tensors.
For each index i from 0 to batchSize - 1, the output is the i-th row of the
the (index[i] + 1)-th tensor.

For i-th row of the output tensor:

y[i][j] = x_{k}[i][j], j = 0,1, ... , (x_{1}.width - 1)

where y is the output tensor. `x_{k}` is the k-th input tensor
and `k = x{0}[i] + 1`.

)DOC");
  }
};

class MultiplexGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase &ctx) const override {
    PADDLE_ENFORCE(!ctx.Inputs("X").empty(), "Input(X) should not be null");
    PADDLE_ENFORCE(!ctx.Outputs(framework::GradVarName("X")).empty(),
                   "Output(X@Grad) should not be null");
    PADDLE_ENFORCE(ctx.HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) shouldn't be null.");
    auto ins = ctx.GetInputsDim("X");
    size_t in_size = ins.size();
    std::vector<framework::DDim> d_ins;
    d_ins.reserve(in_size - 1);
    // don't compute gradient for index(ins[0])
    for (size_t i = 0; i < in_size; i++) {
      d_ins.push_back(ins[i]);
    }
    ctx.SetOutputsDim(framework::GradVarName("X"), d_ins);
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
