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

#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class FlattenOpInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input (X) of Flatten op should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output (Output) of Flatten op should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("XShape"),
                   "Output (XShape) of Flatten op should not be null.");
    const auto &axis = ctx->Attrs().Get<int>("axis");
    const auto &in_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE(axis >= 0, "The axis should be greater than or equal to 0.");
    PADDLE_ENFORCE(
        axis <= in_dims.size(),
        "The axis should be less than or equal to input tensor's rank.");

    const auto &out_dims = GetOutputShape(axis, in_dims);
    ctx->SetOutputDim("XShape", in_dims);
    ctx->ShareLoD("X", "XShape");
    ctx->SetOutputDim("Out", framework::make_ddim(out_dims));
    if (in_dims[0] == out_dims[0]) {
      // Only pass LoD when the first dimension of output and Input(X)
      // are the same.
      ctx->ShareLoD("X", "Out");
    }
  }

  static std::vector<int32_t> GetOutputShape(const int axis,
                                             const framework::DDim &in_dims) {
    int64_t outer = 1, inner = 1;
    for (int i = 0; i < in_dims.size(); ++i) {
      if (i < axis) {
        outer *= in_dims[i];
      } else {
        inner *= in_dims[i];
      }
    }
    std::vector<int32_t> out_shape(2);
    out_shape[0] = outer;
    out_shape[1] = inner;
    return out_shape;
  }
};

class FlattenOp : public framework::OperatorBase {
 public:
  using OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto &axis = Attr<int>("axis");
    auto in_dims =
        scope.FindVar(Input("X"))->Get<framework::LoDTensor>().dims();
    const auto &out_dims = FlattenOpInferShape::GetOutputShape(axis, in_dims);

    framework::AttributeMap attrs;
    attrs["shape"] = out_dims;
    attrs["inplace"] = false;
    // Invoke Reshape Op
    auto reshape_op = framework::OpRegistry::CreateOp(
        "reshape", {{"X", {Input("X")}}, {"Shape", {}}},
        {{"Out", {Output("Out")}}, {"XShape", {Output("XShape")}}}, attrs);
    reshape_op->Run(scope, place);
  }
};

class FlattenOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) A tensor of rank >= axis.");
    AddOutput("Out",
              "A 2D tensor is reshaped input tensor. The input dimensions"
              "up to axis are flattened to the outer dimension of the output"
              "and the remaining input dimensions are flattened into the inner"
              "dimension of the output.");
    AddOutput("XShape",
              "XShape is just used to store the shape and lod of X, which will "
              "be used in FlattenGradOp.")
        .AsIntermediate();
    AddAttr<int>("axis",
                 "(int)"
                 "Indicate up to which input dimensions (exclusive) should be"
                 "flattened to the outer dimension of the output. The value"
                 "for axis must be in the range [0, R], where R is the rank of"
                 "the input tensor. When axis = 0, the shape of the output"
                 "tensor is (1, (d_0 X d_1 ... d_n), where the shape of the"
                 "input tensor is (d_0, d_1, ... d_n).")
        .SetDefault(1);
    AddComment(R"DOC(
Flatten Operator

Flattens the input tensor into a 2D matrix.

Examples:
Case 1:
  Given
    X.shape = (3, 100, 100, 4)
  and
    axis = 2
  We get:
    Out.shape = (3 * 100, 4 * 100)

Case 2:
  Given
    X.shape = (3, 100, 100, 4)
  and
    axis = 0
  We get:
    Out.shape = (1, 3 * 100 * 100 * 4)
)DOC");
  }
};

class FlattenGradOpMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto *grad_op = new framework::OpDesc();
    grad_op->SetType("flatten_grad");
    grad_op->SetInput("XShape", Output("XShape"));
    grad_op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    grad_op->SetAttrMap(Attrs());
    return std::unique_ptr<framework::OpDesc>(grad_op);
  }
};

class FlattenGradInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    context->SetOutputDim(framework::GradVarName("X"),
                          context->GetInputDim("XShape"));
    context->ShareLoD("XShape", framework::GradVarName("X"));
  }
};

class FlattenGradOp : public framework::OperatorBase {
 public:
  using OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto dx_name = Output(framework::GradVarName("X"));
    auto dout_name = Input(framework::GradVarName("Out"));
    auto x_shape = Input("XShape");
    auto x_dims = scope.FindVar(x_shape)->Get<framework::LoDTensor>().dims();

    framework::AttributeMap attrs;
    attrs["shape"] = framework::vectorize2int(x_dims);
    attrs["inplace"] = false;

    auto reshape_op = framework::OpRegistry::CreateOp(
        "reshape", {{"X", {dout_name}}, {"Shape", {}}},
        {{"Out", {dx_name}}, {"XShape", {x_shape}}}, attrs);
    reshape_op->Run(scope, place);
  }
};

}  // namespace operators
}  // namespace paddle

USE_OP(reshape);

namespace ops = paddle::operators;
REGISTER_OPERATOR(flatten, ops::FlattenOp, ops::FlattenOpMaker,
                  ops::FlattenOpInferShape, ops::FlattenGradOpMaker);
REGISTER_OPERATOR(flatten_grad, ops::FlattenGradOp, ops::FlattenGradInferShape);
