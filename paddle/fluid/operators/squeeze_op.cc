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

class SqueezeOpInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of Squeeze operator should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of Squeeze operator should not be null.");

    const auto &x_dims = ctx->GetInputDim("X");
    // Check input tensor dims (<6) Eigen limit.
    PADDLE_ENFORCE(x_dims.size() <= 6,
                   "Invalid dimnesions, the rank of Input(X) "
                   "should be in the range of [1, 6] (Eigen limit).");

    const auto &axes = ctx->Attrs().Get<std::vector<int>>("axes");
    for (int a : axes) {
      PADDLE_ENFORCE_LT(a, x_dims.size(),
                        "The squeeze axis should be less than input "
                        "tensor's rank.");
    }

    auto out_dims = GetOutputShape(axes, x_dims);
    ctx->SetOutputDim("Out", out_dims);
    if (x_dims[0] == out_dims[0]) {
      // Only pass LoD when the first dimension of output and Input(X)
      // are the same.
      ctx->ShareLoD("X", "Out");
    }
  }

  static framework::DDim GetOutputShape(const std::vector<int> squeeze_dims,
                                        const framework::DDim &in_dims) {
    size_t num_squeeze_dims = squeeze_dims.size();
    int cnt_squeezed_dims = 0;
    bool should_squeeze[9] = {false};

    // Determines number of dimensions of output tensor after squeeze.
    // Mark and count the dimensions need to be squeezed
    if (num_squeeze_dims == 0) {
      for (int idx = 0; idx < in_dims.size(); ++idx) {
        if (in_dims[idx] == 1) {
          should_squeeze[idx] = true;
          ++cnt_squeezed_dims;
        }
      }
    } else {
      for (size_t idx = 0; idx < num_squeeze_dims; ++idx) {
        int current = squeeze_dims[idx] < 0 ? squeeze_dims[idx] + in_dims.size()
                                            : squeeze_dims[idx];
        // Check current index, the upper limit has beed checked in line 36.
        PADDLE_ENFORCE(current >= 0,
                       "Invalid axis, the negative axis is out of range.");
        PADDLE_ENFORCE(in_dims[current] == 1,
                       "Invalid axis index, the axis that will be squeezed "
                       "should be equal to 1.");

        if (!(should_squeeze[current])) {
          ++cnt_squeezed_dims;
        }
        should_squeeze[current] = true;
      }
    }

    // Make output dimensions
    std::vector<int64_t> output_shape(in_dims.size() - cnt_squeezed_dims, 0);
    for (int in_idx = 0, out_idx = 0; in_idx < in_dims.size(); ++in_idx) {
      if (!should_squeeze[in_idx]) {
        output_shape[out_idx++] = in_dims[in_idx];
      }
    }

    return framework::make_ddim(output_shape);
  }
};

// TODO(paddle-dev): Should use OpKernel.
class SqueezeOp : public framework::OperatorBase {
 public:
  using OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto &axes = Attr<std::vector<int>>("axes");
    auto x_dims = scope.FindVar(Input("X"))->Get<framework::LoDTensor>().dims();
    auto out_dims = SqueezeOpInferShape::GetOutputShape(axes, x_dims);

    framework::AttributeMap attrs;
    attrs["shape"] = framework::vectorize2int(out_dims);
    // Invoke Reshape Op
    auto reshape_op = framework::OpRegistry::CreateOp(
        "reshape", {{"X", {Input("X")}}, {"Shape", {}}},
        {{"Out", {Output("Out")}}}, attrs);
    reshape_op->Run(scope, place);
  }
};

class SqueezeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor). The input tensor of squeeze operator.");
    AddOutput("Out", "(Tensor). The output tensor of squeeze operator.");
    AddAttr<std::vector<int>>("axes",
                              "(std::vector<int>). List of integers,"
                              " indicating the dimensions to squeeze.")
        .SetDefault({});
    AddComment(R"DOC(
        Squeeze Operator.

        Remove single-dimensional entries from the shape of a tensor.
        Takes a parameter axes with a list of axes to squeeze.
        If axes is not provided, all the single dimensions will be removed from the shape.
        If an axis is selected with shape entry not equal to one, an error is raised.

        Examples:
        Case 1:
          Given
            X.shape = (1, 3, 1, 5)
          and
            axes = [0]
          we get:
            Out.shape = (3, 1, 5)

        Case 2:
          Given
            X.shape = (1, 3, 1, 5)
          and
            axes = []
          we get:
            Out.shape = (3, 5)
    )DOC");
  }
};

class SqueezeGradInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    context->SetOutputDim(framework::GradVarName("X"),
                          context->GetInputDim("X"));
    context->ShareLoD("X", framework::GradVarName("X"));
  }
};

class SqueezeGradOp : public framework::OperatorBase {
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

// FIXME(zcd): squeeze2 adds an intermediate output(XShape) based on squeeze,
// the XShape is used to carry the shape and lod of X which will be used in
// squeeze_grad, in this way, the framework can reuse the memory of X
// immediately the squeeze2_op is finished.
// Considering compatibility issues, we could not fix squeeze2_op
class Squeeze2OpMaker : public SqueezeOpMaker {
 public:
  void Make() override {
    SqueezeOpMaker::Make();
    AddOutput("XShape",
              "XShape is just used to store the shape and lod of X, which will "
              "be used in SqueezeGradOp.")
        .AsIntermediate();
  }
};

class Squeeze2OpInferShape : public SqueezeOpInferShape {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    SqueezeOpInferShape::operator()(ctx);
    PADDLE_ENFORCE(ctx->HasOutput("XShape"),
                   "Output(XShape) of Squeeze operator should not be null.");
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

class Squeeze2Op : public framework::OperatorBase {
 public:
  using OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto &axes = Attr<std::vector<int>>("axes");
    auto x_dims = scope.FindVar(Input("X"))->Get<framework::LoDTensor>().dims();
    auto out_dims = Squeeze2OpInferShape::GetOutputShape(axes, x_dims);

    framework::AttributeMap attrs;
    attrs["shape"] = framework::vectorize2int(out_dims);
    // Invoke Reshape Op
    auto reshape_op = framework::OpRegistry::CreateOp(
        "reshape2", {{"X", {Input("X")}}, {"Shape", {}}},
        {{"Out", {Output("Out")}}, {"XShape", {Output("XShape")}}}, attrs);
    reshape_op->Run(scope, place);
  }
};

class Squeeze2GradOpMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto *grad_op = new framework::OpDesc();
    grad_op->SetType("squeeze2_grad");
    grad_op->SetInput("XShape", Output("XShape"));
    grad_op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    grad_op->SetAttrMap(Attrs());
    return std::unique_ptr<framework::OpDesc>(grad_op);
  }
};

class Squeeze2GradInferShape : public framework::InferShapeBase {
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

class Squeeze2GradOp : public framework::OperatorBase {
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

// Tell linker to use reshape op
USE_OP(reshape);

namespace ops = paddle::operators;
REGISTER_OPERATOR(squeeze, ops::SqueezeOp, ops::SqueezeOpMaker,
                  ops::SqueezeOpInferShape,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(squeeze_grad, ops::SqueezeGradOp, ops::SqueezeGradInferShape);

REGISTER_OPERATOR(squeeze2, ops::Squeeze2Op, ops::Squeeze2OpMaker,
                  ops::Squeeze2OpInferShape, ops::Squeeze2GradOpMaker);
REGISTER_OPERATOR(squeeze2_grad, ops::Squeeze2GradOp,
                  ops::Squeeze2GradInferShape);
