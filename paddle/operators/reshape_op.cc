
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

#include "paddle/operators/reshape_op.h"

namespace paddle {
namespace operators {

class ReshapeOp : public framework::OperatorWithKernel {
 public:
  ReshapeOp(const std::string &type, const framework::VariableNameMap &inputs,
            const framework::VariableNameMap &outputs,
            const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    // input check
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of ReshapeOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of ReshapeOp should not be null.");

    auto shape = ctx->Attrs().Get<std::vector<int>>("shape");
    PADDLE_ENFORCE(shape.size() > 0, "Attr(shape) shouldn't be empty.");
    auto x_dims = ctx->GetInputDim("X");
    // TODO(qiao) change batch_size
    for (size_t i = 1; i < shape.size(); ++i) {
      PADDLE_ENFORCE(shape[i] > 0,
                     "Each dimension of shape "
                     "must be positiv except the first.");
    }
    if (shape[0] < 0) {
      shape[0] = x_dims[0];
    }
    // capacity check
    int64_t capacity =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    int64_t in_size = framework::product(x_dims);
    PADDLE_ENFORCE_EQ(capacity, in_size,
                      "The size of Input(X) mismatches with Attr(shape).");
    // resize output
    std::vector<int64_t> shape_int64(shape.size(), 0);
    std::transform(shape.begin(), shape.end(), shape_int64.begin(),
                   [](int a) { return static_cast<int64_t>(a); });
    auto out_dims = framework::make_ddim(shape_int64);
    ctx->SetOutputDim("Out", out_dims);
    if (shape[0] == x_dims[0]) {
      // Only pass LoD when the first dimension is equal between
      // output and input.
      ctx->ShareLoD("X", /*->*/ "Out");
    }
  }
};

class ReshapeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ReshapeOpMaker(framework::OpProto *proto,
                 framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input tensor of reshape operator.");
    AddOutput("Out", "The output tensor of reshape operator.");
    AddAttr<std::vector<int>>("shape",
                              "(vector<int>) "
                              "Target shape of reshape operator.");
    AddComment(R"DOC(
Reshape Operator.

Reshape Input(X) into the shape specified by Attr(shape).

An example:
Given a 2-D tensor X with 2 rows and 2 columns

    [[1, 2], [3, 4]]

and target shape = [1, 4], the reshape operator will transform
the tensor X into a 1-D tensor:

    [1, 2, 3, 4]

)DOC");
  }
};

class ReshapeGradOp : public framework::OperatorWithKernel {
 public:
  ReshapeGradOp(const std::string &type,
                const framework::VariableNameMap &inputs,
                const framework::VariableNameMap &outputs,
                const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) shouldn't be null.");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OP(reshape, ops::ReshapeOp, ops::ReshapeOpMaker, reshape_grad,
            ops::ReshapeGradOp);
REGISTER_OP_CPU_KERNEL(reshape,
                       ops::ReshapeKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    reshape_grad, ops::ReshapeGradKernel<paddle::platform::CPUPlace, float>);
