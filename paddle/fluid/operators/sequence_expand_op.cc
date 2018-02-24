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

#include "paddle/fluid/operators/sequence_expand_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class SequenceExpandOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"));
    PADDLE_ENFORCE(ctx->HasOutput("Out"));
    PADDLE_ENFORCE(ctx->HasInput("Y"));
    framework::DDim out_dim;
    auto y_dim = ctx->GetInputDim("Y");
    out_dim = ctx->GetInputDim("X");
    out_dim[0] = y_dim[0];
    ctx->ShareLoD("Y", "Out");
    ctx->SetOutputDim("Out", out_dim);
  }
};

class SequenceExpandOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SequenceExpandOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "(Tensor or LoDTensor) The input(X) of this operator can be a "
             "LoDTensor or a base Tensor.");
    AddInput("Y",
             "(LoDTensor)The reference input(Y) of sequence_expand op."
             "It must be a LoDTensor with k-level(k>0)."
             "The input(X) will be expanded according to LOD of input(Y)."
             "The element numbers of last level in input(Y) "
             "must be equal to dims[0] of input(X).");
    AddOutput("Out",
              "(LodTensor)The output of sequence_expand op."
              "The lod of output will be as same as input(Y)'s lod.");
    AddComment(R"DOC(
Sequence Expand Operator.

This operator expands input(X) according to LOD of input(Y).
Following are cases to better explain how this works:
Case 1:

Given a 2-level LoDTensor input(X)
    X.lod = [[0,       2, 3],
             [0, 1,    3, 4]]
    X.data = [a, b, c, d]
    X.dims = [4, 1]
and input(Y)
    Y.lod = [[0,    2,    4],
             [0, 3, 6, 7, 8]]
with condition len(Y.lod[-1]) -1 == X.dims[0]
then we get 2-level LoDTensor
    Out.lod = [[0,                2,    4],
               [0,       3,       6, 7, 8]]
    Out.data = [a, a, a, b, b, b, c, d]
    Out.dims = [8, 1]

Case 2:

Given a common Tensor input(X)
    X.data = [a, b, c]
    X.dims = [3, 1]
and input(Y)
    Y.lod = [[0, 2, 3, 6]]
with condition len(Y.lod[-1]) -1 == X.dims[0]
then we get 1-level LoDTensor
    Out.lod = [[0,    2, 3,      6]]
    Out.data = [a, a, b, c, c, c]
    Out.dims = [6, 1]

Case 3:

Given a common Tensor input(X)
    X.data = [[a, b], [c, d], [e, f]]
    X.dims = [3, 2]
and input(Y)
    Y.lod = [[0, 2, 3, 6]]
with condition len(Y.lod[-1]) -1 == X.dims[0]
then we get 1-level LoDTensor
    Out.lod = [[0,           2,     3,                     6]]
    Out.data = [[a,b], [a,b] [c,d], [e, f], [e, f], [e, f]]
    Out.dims = [6, 2]

Case 4:

Given 2-level a LoDTensor input(X)
    X.lod = [[0,       2, 3],
             [0, 1,    3, 4]]
    X.data = [a, b, c, d]
    X.dims = [4, 1]
and input(Y)
    Y.lod = [[0,    2,    4],
             [0, 3, 6, 6, 8]]
with condition len(Y.lod[-1]) -1 == X.dims[0]
then we get 2-level LoDTensor
    Out.lod = [[0,                2,    4],
               [0,       3,       6, 6, 8]]
    Out.data = [a, a, a, b, b, b, d, d]
    Out.dims = [8, 1]


)DOC");
  }
};

class SequenceExpandOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"));
    PADDLE_ENFORCE(ctx->HasInput("Out"));
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "The input(Out@GRAD) should not be null");
    auto x_dims = ctx->GetInputDim("X");
    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(sequence_expand, ops::SequenceExpandOp, ops::SequenceExpandOpMaker,
            sequence_expand_grad, ops::SequenceExpandOpGrad);
REGISTER_OP_CPU_KERNEL(
    sequence_expand,
    ops::SequenceExpandKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    sequence_expand_grad,
    ops::SequenceExpandGradKernel<paddle::platform::CPUDeviceContext, float>);
