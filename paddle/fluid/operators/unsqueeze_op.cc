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

#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class UnsqueezeOpInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of Unsqueeze operator should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of Unsqueeze operator should not be null.");

    const auto &axes = ctx->Attrs().Get<std::vector<int>>("axes");
    const auto &x_dims = ctx->GetInputDim("X");
    // Validity Check: input tensor dims (<6).
    PADDLE_ENFORCE(x_dims.size() <= 6,
                   "Invalid dimensions, the rank of Input(X) "
                   "should be in the range of [1, 6] (Eigen limit)");
    auto out_dims = GetOutputShape(axes, x_dims);
    ctx->SetOutputDim("Out", out_dims);
    if (x_dims[0] == out_dims[0]) {
      // Only pass LoD when the first dimension of output and Input(X)
      // are the same.
      ctx->ShareLoD("X", "Out");
    }
  }

  static framework::DDim GetOutputShape(const std::vector<int> unsqz_dims,
                                        const framework::DDim &in_dims) {
    int output_size = in_dims.size() + static_cast<int>(unsqz_dims.size());
    int cur_output_size = in_dims.size();
    std::vector<int64_t> output_shape(output_size, 0);

    // Validity Check: rank range.
    PADDLE_ENFORCE(output_size <= 6,
                   "The output tensor's rank should be less than 6.");

    for (int axis : unsqz_dims) {
      int cur = axis < 0 ? axis + cur_output_size + 1 : axis;
      // Vaildity Check: the axis bound
      PADDLE_ENFORCE(
          cur >= 0 && cur <= cur_output_size,
          "The unsqueeze dims must be within range of current rank.");
      // Move old axis, and insert new axis
      for (int i = cur_output_size; i >= cur; --i) {
        if (output_shape[i] == 1) {
          // Move axis
          output_shape[i + 1] = 1;
          output_shape[i] = 0;
        }
      }
      output_shape[cur] = 1;
      // Add the output size.
      cur_output_size++;
    }

    // Make output shape
    for (int in_idx = 0, out_idx = 0; out_idx < output_size; ++out_idx) {
      if (output_shape[out_idx] == 0) {
        output_shape[out_idx] = in_dims[in_idx++];
      }
    }

    return framework::make_ddim(output_shape);
  }
};

class UnsqueezeOp : public framework::OperatorBase {
 public:
  using OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto &axes = Attr<std::vector<int>>("axes");
    auto x_dims = scope.FindVar(Input("X"))->Get<framework::LoDTensor>().dims();
    auto out_dims = UnsqueezeOpInferShape::GetOutputShape(axes, x_dims);

    framework::AttributeMap attrs;
    attrs["shape"] = framework::vectorize2int(out_dims);
    // Invoke Reshape op.
    auto reshape_op = framework::OpRegistry::CreateOp(
        "reshape", {{"X", {Input("X")}}, {"Shape", {}}},
        {{"Out", {Output("Out")}}}, attrs);
    reshape_op->Run(scope, place);
  }
};

class UnsqueezeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor). The input tensor of unsqueeze operator.");
    AddOutput("Out", "(Tensor). The output tensor of unsqueeze operator.");
    AddAttr<std::vector<int>>("axes",
                              "(std::vector<int>). List of integers,"
                              " indicating the dimensions to be inserted")
        .AddCustomChecker([](const std::vector<int> &axes) {
          PADDLE_ENFORCE(!axes.empty(),
                         "Invalid axes, The unsqueeze axes is empty.");
          // Validity Check: axes dims (<6).
          PADDLE_ENFORCE(static_cast<int>(axes.size()) < 6,
                         "Invalid dimensions, dynamic dimensions should be "
                         "within [1, 6] dimensions (Eigen limit).");
          // Validity Check: the range of unsqueeze aixs.
          for (int axis : axes) {
            PADDLE_ENFORCE(axis < 6,
                           "Invalid dimensions, input axis should be"
                           " within [1, 6] dimensions (Eigen limit).");
          }
        });
    AddComment(R"DOC(
    Unsqueeze Operator.

    Insert single-dimensional entries to the shape of a tensor.
    Takes one required argument axes, a list of dimensions that will be inserted.
    Dimension indices in axes are as seen in the output tensor.

    For example:
      Given a tensor such that tensor with shape [3, 4, 5],
      then Unsqueeze(tensor, axes=[0, 4]) has shape [1, 3, 4, 5, 1]
    )DOC");
  }
};

class UnsqueezeGradInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    ctx->ShareLoD("X", framework::GradVarName("X"));
  }
};

class UnsqueezeGradOp : public framework::OperatorBase {
 public:
  using OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto dx_name = Output(framework::GradVarName("X"));
    auto dout_name = Input(framework::GradVarName("Out"));
    auto x_dims = scope.FindVar(Input("X"))->Get<framework::LoDTensor>().dims();

    framework::AttributeMap attrs;
    attrs["shape"] = framework::vectorize2int(x_dims);

    auto reshape_op = framework::OpRegistry::CreateOp(
        "reshape", {{"X", {dout_name}}, {"Shape", {}}}, {{"Out", {dx_name}}},
        attrs);
    reshape_op->Run(scope, place);
  }
};

// FIXME(zcd): unsqueeze2 adds an intermediate output(XShape) based on
// unsqueeze, the XShape is used to carry the shape and lod of X which
// will be used in unsqueeze_grad, in this way, the framework can reuse
// the memory of X immediately the unsqueeze2_op is finished.
// Considering compatibility issues, we could not fix unsqueeze2_op
class Unsqueeze2OpInferShape : public UnsqueezeOpInferShape {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    UnsqueezeOpInferShape::operator()(ctx);
    PADDLE_ENFORCE(ctx->HasOutput("XShape"),
                   "Output(XShape) of Unsqueeze operator should not be null.");
    const auto &x_dims = ctx->GetInputDim("X");
    std::vector<int64_t> xshape_dims(x_dims.size() + 1);
    xshape_dims[0] = 0;
    for (int i = 0; i < x_dims.size(); ++i) {
      xshape_dims[i + 1] = x_dims[i];
    }
    ctx->SetOutputDim("XShape", framework::make_ddim(xshape_dims));
    ctx->ShareLoD("X", /*->*/ "XShape");
  }
};

class Unsqueeze2OpMaker : public UnsqueezeOpMaker {
 public:
  void Make() override {
    UnsqueezeOpMaker::Make();
    AddOutput("XShape",
              "XShape is just used to store the shape and lod of X, which will "
              "be used in UnsqueezeGradOp.")
        .AsIntermediate();
  }
};

class Unsqueeze2Op : public framework::OperatorBase {
 public:
  using OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto &axes = Attr<std::vector<int>>("axes");
    auto x_dims = scope.FindVar(Input("X"))->Get<framework::LoDTensor>().dims();
    auto out_dims = Unsqueeze2OpInferShape::GetOutputShape(axes, x_dims);

    framework::AttributeMap attrs;
    attrs["shape"] = framework::vectorize2int(out_dims);
    // Invoke Reshape op.
    auto reshape_op = framework::OpRegistry::CreateOp(
        "reshape2", {{"X", {Input("X")}}, {"Shape", {}}},
        {{"Out", {Output("Out")}}, {"XShape", {Output("XShape")}}}, attrs);
    reshape_op->Run(scope, place);
  }
};

class Unsqueeze2GradOpMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto *grad_op = new framework::OpDesc();
    grad_op->SetType("unsqueeze2_grad");
    grad_op->SetInput("XShape", Output("XShape"));
    grad_op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    grad_op->SetAttrMap(Attrs());
    return std::unique_ptr<framework::OpDesc>(grad_op);
  }
};

class Unsqueeze2GradInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE(context->HasInput("XShape"),
                   "Input(XShape) shouldn't be null.");
    PADDLE_ENFORCE(context->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) shouldn't be null.");
    auto xshape_dims = context->GetInputDim("XShape");
    auto x_dims = framework::slice_ddim(xshape_dims, 1, xshape_dims.size());
    context->SetOutputDim(framework::GradVarName("X"), x_dims);
    context->ShareLoD("XShape", framework::GradVarName("X"));
  }
};

class Unsqueeze2GradOp : public framework::OperatorBase {
 public:
  using OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto dx_name = Output(framework::GradVarName("X"));
    auto dout_name = Input(framework::GradVarName("Out"));
    auto xshape_name = Input("XShape");
    auto xshape_dims =
        scope.FindVar(xshape_name)->Get<framework::LoDTensor>().dims();
    auto x_dims = framework::slice_ddim(xshape_dims, 1, xshape_dims.size());

    framework::AttributeMap attrs;
    attrs["shape"] = framework::vectorize2int(x_dims);

    auto reshape_op = framework::OpRegistry::CreateOp(
        "reshape2", {{"X", {dout_name}}, {"Shape", {}}},
        {{"Out", {dx_name}}, {"XShape", {xshape_name}}}, attrs);
    reshape_op->Run(scope, place);
  }
};
}  // namespace operators
}  // namespace paddle

// Tell linker to use reshape op.
USE_OP(reshape);

namespace ops = paddle::operators;
REGISTER_OPERATOR(unsqueeze, ops::UnsqueezeOp, ops::UnsqueezeOpMaker,
                  ops::UnsqueezeOpInferShape,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(unsqueeze_grad, ops::UnsqueezeGradOp,
                  ops::UnsqueezeGradInferShape);

REGISTER_OPERATOR(unsqueeze2, ops::Unsqueeze2Op, ops::Unsqueeze2OpMaker,
                  ops::Unsqueeze2OpInferShape, ops::Unsqueeze2GradOpMaker);
REGISTER_OPERATOR(unsqueeze2_grad, ops::Unsqueeze2GradOp,
                  ops::Unsqueeze2GradInferShape);
