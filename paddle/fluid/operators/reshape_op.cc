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

    const std::vector<int> &shape = ctx->Attrs().Get<std::vector<int>>("shape");
    PADDLE_ENFORCE_EQ(shape.empty(), ctx->HasInput("Shape"),
                      "The shape information can only be set by Attr(shape) or "
                      "by Input(Shape). Attr(shape) and Input(Shape) cannot be "
                      "set at the same time.");

    auto x_dims = ctx->GetInputDim("X");

    if (ctx->HasInput("Shape")) {
      // The shape information in given by Input(Shape).
      auto shape_dims = ctx->GetInputDim("Shape");

      PADDLE_ENFORCE(shape_dims.size() == 2UL && shape_dims[0] == 1UL,
                     "The Input(Label) should be a 2-D tensor with the 1st "
                     "dimensions fixed to 1 (a row vector).");

      // The actual output shape will be set at runtime, here temporially set
      // the shape of output the same as the shape of input.
      ctx->SetOutputDim("Out", x_dims);
    } else {
      // The shape information in given by Attr(shape).
      std::vector<int64_t> output_shape;
      ValidateShape(shape, framework::product(x_dims), output_shape);

      auto out_dims = framework::make_ddim(output_shape);
      ctx->SetOutputDim("Out", out_dims);

      if (shape[0] == x_dims[0]) {
        // Only pass LoD when the first dimension of output and Input(X)
        // are the same.
        ctx->ShareLoD("X", /*->*/ "Out");
      }
    }
  }

 private:
  void ValidateShape(const std::vector<int> &shape, const int64_t in_size,
                     std::vector<int64_t> &output_shape) const {
    std::vector<size_t> neg_dims_idx;
    const int unknown_index = -1;  // only one dimension canbe set to -1, whose
                                   // size will be automatically infered.

    for (size_t i = 0; i < shape.size(); ++i) {
      PADDLE_ENFORCE(shape[i] > 1 || shape[i] == unknown_index,
                     "Each input dimension of Attr(shape) must be positive, or "
                     "only one input dimension can be -1.");
      if (shape[i] == unknown_index) neg_dims_idx.push_back(i);
    }
    PADDLE_ENFORCE_LE(
        neg_dims_idx.size(), 1,
        "Only one input dimension of Attr(shape) may be unknown.");

    int64_t inferred_dim = 0;
    if (neg_dims_idx.size()) {
      int64_t capacity = std::accumulate(shape.begin(), shape.end(), 1,
                                         std::multiplies<int>());
      inferred_dim = in_size / (-capacity);
    }

    output_shape.resize(shape.size(), 0);
    std::transform(shape.begin(), shape.end(), output_shape.begin(),
                   [](int a) { return static_cast<int64_t>(a); });
    if (neg_dims_idx.size()) output_shape[neg_dims_idx[0]] = inferred_dim;
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<framework::Tensor>("X")->type()),
        ctx.device_context());
  }
};

class ReshapeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ReshapeOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input tensor of reshape operator.");
    AddInput(
        "Shape",
        "Tensor<int64_t>, a 1-D tensor that provides the shape information.")
        .AsDispensable();
    AddOutput("Out", "The output tensor of reshape operator.");
    AddAttr<std::vector<int>>(
        "shape", "(std::vector<int>) Target shape of reshape operator.")
        .SetDefault(std::vector<int>());
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

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<framework::Tensor>("X")->type()),
        ctx.device_context());
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
