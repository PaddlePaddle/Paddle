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

#include "paddle/operators/seq_expand_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class SeqExpandOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of SeqExpandOp should not be null.");
    int repeat = ctx->Attrs().Get<int>("repeat");
    framework::DDim out_dim;
    if (repeat == 0) {
      PADDLE_ENFORCE(
          ctx->HasInput("Y"),
          "Input(Y) of SeqExpandOp should not be null while repeat == 0.");
      out_dim = ctx->GetInputDim("Y");
      ctx->ShareLoD("Y", "Out");
    } else {
      out_dim = ctx->GetInputDim("X");
      out_dim[0] = out_dim[0] * repeat;
    }
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of PadOp should not be null.");
    ctx->SetOutputDim("Out", out_dim);
  }
};

class SeqExpandOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SeqExpandOpMaker(framework::OpProto* proto,
                   framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(
        "X",
        "The input('X') of seq_expand op. It can be LoDTensor or base Tensor.");
    AddInput(
        "Y",
        "The reference input('Y') of seq_expand op."
        "It must be a LoDTensor with k-level(k>0)."
        "This reference input is essential if 'repeat' attribute is not "
        "configured."
        "Input(X) will be expanded by LoD of input(Y) while repeat ==  0.");
    AddOutput("Out",
              "The output of seq_expand op."
              "The output is a (k+1)-level LoDTensor"
              "while input(X) being k-level LoDTensor."
              "(Given base tensor is 0-level LoDTensor.)");
    AddAttr<int>("repeat",
                 "(type:int; default value: 0)"
                 "Repeatting times of each element while expanding input(X)."
                 "It works while input(Y) is not configured.")
        .SetDefault(0);
    AddComment(R"DOC(
Expand k-level LoDTensor to (k+1)-level LoDTensor
by lod of input(Y) or 'repeat' attribute.

Case 1:

Given a 2-level LoDTensor X:
    X.data = [1, 2 , 3, 4]
    X.lod = [[0, 3, 4], [0, 1, 3, 4]]
and
    repeat = 2
then we get 3-level LoDTensor
    Out.data = [1, 2, 3, 1, 2, 3, 4, 4]
    Out.lod = [[0, 6, 8],
               [0, 3, 6, 7, 8],
               [0, 1, 3, 4, 6, 7, 8]]

Case 2:

Given 2-level a LoDTensor X
    X.data = [1, 2, 3, 4]
    X.lod = [[0, 3, 4], [0, 1, 3, 4]]
and
    Y.lod = [[0, 6, 8],
             [0, 3, 6, 7, 8],
             [0,1,3,4,6,7,8]]
then we get 3-level LoDTensor
    Out.data = [1, 2, 3, 1, 2, 3, 4, 4]
    Out.lod = [[0, 6, 8],
               [0, 3, 6, 7, 8],
               [0, 1, 3, 4, 6, 7, 8]]

Case 3:

Given a 0-level LoDTensor X
    X.data = [1, 2, 3, 4]
    X.lod = NULL
and
    repeat = 2
then we get 1-level LoDTensor
    Out.data = [1, 1, 2, 2, 3, 3, 4, 4]
    Out.lod = [[0, 2, 4, 6, 8]]

)DOC");
  }
};

class SeqExpandOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasInput("Out"), "Input(Out) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");
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
REGISTER_OP(seq_expand, ops::SeqExpandOp, ops::SeqExpandOpMaker,
            seq_expand_grad, ops::SeqExpandOpGrad);
REGISTER_OP_CPU_KERNEL(seq_expand,
                       ops::SeqExpandKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    seq_expand_grad,
    ops::SeqExpandGradKernel<paddle::platform::CPUPlace, float>);
