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
  MultiplexOp(const std::string &type, const framework::VariableNameMap &inputs,
              const framework::VariableNameMap &outputs,
              const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE(!ctx.MultiInputVar("X").empty(),
                   "Input(X) should not be null");
    PADDLE_ENFORCE_NOT_NULL(ctx.OutputVar("Out"),
                            "Output(Out) shouldn't be null.");
    auto ins = ctx.MultiInput<Tensor>("X");
    auto *out = ctx.Output<LoDTensor>("Out");
    auto num_ins = ins.size();
    PADDLE_ENFORCE(num_ins > 2,
                   "multiplex operator should have more than 2 inputs.");
    PADDLE_ENFORCE_EQ(ins[0]->dims().size(), 1,
                      "The first input must be a index vector.");
    auto in_dim = ins[1]->dims();

    for (size_t i = 2; i < num_ins; i++) {
      auto dim = ins[i]->dims();
      PADDLE_ENFORCE(
          in_dim == dim,
          "All the input tensors except the first one must have the same size");
    }
    out->Resize(in_dim);
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

ins[0]: the index of the tensor to output of size batchSize.
ins[1:N]: the candidate output tensor.
For each index i from 0 to batchSize - 1, the output is the i-th row of the
the (index[i] + 1)-th tensor.

For each i-th row of output:

y[i][j] = x_{k}[i][j], j = 0,1, ... , (x_{1}.width - 1)

where y is the output tensor. `x_{k}` is the k-th input tensor
and `k = x{0}[i] + 1`.

)DOC");
  }
};

class MultiplexGradOp : public framework::OperatorWithKernel {
 public:
  MultiplexGradOp(const std::string &type,
                  const framework::VariableNameMap &inputs,
                  const framework::VariableNameMap &outputs,
                  const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE(!ctx.MultiInputVar("X").empty(),
                   "Input(X) should not be null");
    PADDLE_ENFORCE(!ctx.MultiOutputVar(framework::GradVarName("X")).empty(),
                   "Output(X@Grad) should not be null");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar(framework::GradVarName("Out")),
                            "Input(Out@GRAD) shouldn't be null.");
    auto d_ins = ctx.MultiOutput<LoDTensor>(framework::GradVarName("X"));
    auto ins = ctx.MultiInput<Tensor>("X");
    // don;t compute gradient for index
    for (size_t i = 1; i < ins.size(); i++) {
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
REGISTER_OP_CPU_KERNEL(multiplex, ops::MultiplexCPUKernel<float>);
REGISTER_OP_CPU_KERNEL(multiplex_grad, ops::MultiplexGradCPUKernel<float>);
