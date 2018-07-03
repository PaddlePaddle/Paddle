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

#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class UnsqueezeOpInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of UnsqueezeOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of UnsqueezeOp should not be null.");

    const auto &axes = ctx->Attrs().Get<std::vector<int>>("axes");
    PADDLE_ENFORCE(!axes.empty(),
                   "The unsqueeze axes information must be set by Attr(axes).");

    const auto &x_dims = ctx->GetInputDim("X");
    // Validity Check: input tensor dims (<6).
    PADDLE_ENFORCE(static_cast<int>(x_dims.size()) <= 6,
                   "Invalid dimensions, dynamic dimensions should within "
                   "[1, 6] dimensions (Eigen limit).");
    // Validity Check: the range of unsqueeze aixs.
    for (int axis : axes) {
      PADDLE_ENFORCE(axis < 6,
                     "Invalid dimensions, input axis should within "
                     "[1, 6] dimensions (Eigen limit).");
    }

    auto out_dims = GetOutputShape(axes, x_dims);
    ctx->SetOutputDim("Out", out_dims);
  }

  static framework::DDim GetOutputShape(const std::vector<int> unsqz_dims,
                                        const framework::DDim &in_dims) {
    unsigned int unsqz_mask = 0;
    unsigned int front = 0, back = 0;
    int output_dims_size = in_dims.size();

    // Simulate insert by bit calc.
    for (int axis : unsqz_dims) {
      int cur = axis < 0 ? axis + output_dims_size + 1 : axis;
      // Vaildity Check: the axis bound
      PADDLE_ENFORCE(
          cur >= 0 && cur <= output_dims_size,
          "The unsqueeze dims must be within range of current rank.");
      // Save the front part.
      front = unsqz_mask & ((1 << cur) - 1);
      // Move the back part.
      back = unsqz_mask & ~((1 << cur) - 1);
      back <<= 1;
      // Merge two part.
      back |= (1 << cur);
      unsqz_mask = front | back;
      // Add the output size.
      output_dims_size++;
      // Validity Check: rank range.
      PADDLE_ENFORCE(output_dims_size <= 6,
                     "The output tensor's rank should be less than 6.");
    }

    // Make output shape
    std::vector<int64_t> output_shape(output_dims_size, 0);
    for (int in_idx = 0, out_idx = 0; out_idx < output_dims_size; ++out_idx) {
      if ((unsqz_mask & (1 << out_idx)) == 0) {
        output_shape[out_idx] = in_dims[in_idx++];
      } else {
        output_shape[out_idx] = 1;
      }
    }

    return framework::make_ddim(output_shape);
  }
};

class UnsqueezeOp : public framework::OperatorBase {
 public:
  UnsqueezeOp(const std::string &type, const framework::VariableNameMap &inputs,
              const framework::VariableNameMap &outputs,
              const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto &axes = Attr<std::vector<int>>("axes");
    auto x_dims = scope.FindVar(Input("X"))->Get<framework::LoDTensor>().dims();
    auto out_dims = UnsqueezeOpInferShape::GetOutputShape(axes, x_dims);

    framework::AttributeMap attrs;
    attrs["shape"] = framework::vectorize2int(out_dims);
    attrs["inplace"] = Attr<bool>("inplace");
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
                              "(std::vector<int>). List of positive integers,"
                              " indicate the dimensions to be inserted");
    AddAttr<bool>(
        "inplace",
        "(default: false) Unsqueeze the source tensor's shape without "
        "memory copy. When Attr(inplace) is set true, the output "
        "tensor shares memory with Input(X), otherwise, a new output "
        "tensor is created, and its data are copied from Input(x).")
        .SetDefault(false);
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
  UnsqueezeGradOp(const std::string &type,
                  const framework::VariableNameMap &inputs,
                  const framework::VariableNameMap &outputs,
                  const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto dx_name = Output(framework::GradVarName("X"));
    auto dout_name = Input(framework::GradVarName("Out"));
    auto x_dims = scope.FindVar(Input("X"))->Get<framework::LoDTensor>().dims();

    framework::AttributeMap attrs;
    attrs["shape"] = framework::vectorize2int(x_dims);
    attrs["inplace"] = Attr<bool>("inplace");

    auto reshape_op = framework::OpRegistry::CreateOp(
        "reshape", {{"X", {dout_name}}, {"Shape", {}}}, {{"Out", {dx_name}}},
        attrs);
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
