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

#include "paddle/fluid/operators/reshape_op.h"

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

    std::vector<size_t> neg_dims_idx;
    // set some dimension to -1 if it is unknown
    const int unknown_size = -1;
    for (size_t i = 0; i < shape.size(); ++i) {
      PADDLE_ENFORCE(shape[i] > 0 || shape[i] == unknown_size,
                     "Each dimension of Attr(shape) must be positive or %d.",
                     unknown_size);
      if (shape[i] == unknown_size) {
        neg_dims_idx.push_back(i);
        PADDLE_ENFORCE(neg_dims_idx.size() <= 1,
                       "Only one dimension of Attr(shape) can be unknown.");
      }
    }

    int64_t capacity =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    int64_t in_size = framework::product(x_dims);
    if (neg_dims_idx.size() == 1) {
      // dim infer
      shape[neg_dims_idx[0]] = in_size / (-capacity);
      // recalculate capacity
      capacity = shape[neg_dims_idx[0]] * (-capacity);
    }
    // capacity check
    PADDLE_ENFORCE(capacity == in_size,
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
  ReshapeOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input tensor of reshape operator.");
    AddOutput("Out", "The output tensor of reshape operator.");
    AddAttr<std::vector<int>>("shape",
                              "(vector<int>) "
                              "Target shape of reshape operator.");
    AddAttr<bool>("inplace",
                  "Change the source tensor's shape without copy memory.")
        .SetDefault(true);
    AddComment(R"DOC(
Reshape Operator.

Reshape Input(X) into the shape specified by Attr(shape).

An example:
Given a 2-D tensor X with 2 rows and 2 columns : [[1, 2], [3, 4]]

and target shape = [1, 4], the reshape operator will transform
the tensor X into a 2-D tensor: [[1, 2, 3, 4]]

One dimension in the target shape can be set -1, representing that its
size is unknown. In this case, the real dimension will be infered from 
the original shape of Input(X) and other dimensions in the target shape.
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
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OP(reshape, ops::ReshapeOp, ops::ReshapeOpMaker, reshape_grad,
            ops::ReshapeGradOp);
REGISTER_OP_CPU_KERNEL(reshape, ops::ReshapeKernel<CPU, float>,
                       ops::ReshapeKernel<CPU, double>,
                       ops::ReshapeKernel<CPU, int>,
                       ops::ReshapeKernel<CPU, int64_t>);
REGISTER_OP_CPU_KERNEL(reshape_grad, ops::ReshapeGradKernel<CPU, float>,
                       ops::ReshapeGradKernel<CPU, double>,
                       ops::ReshapeGradKernel<CPU, int>,
                       ops::ReshapeGradKernel<CPU, int64_t>);
