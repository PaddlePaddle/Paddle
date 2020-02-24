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

#include "paddle/fluid/operators/pad_op.h"
#include <memory>

namespace paddle {
namespace operators {

using framework::Tensor;

class PadOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) of PadOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of PadOp should not be null.");

    auto x_dim = ctx->GetInputDim("X");
    auto& paddings = ctx->Attrs().Get<std::vector<int>>("paddings");
    PADDLE_ENFORCE_EQ(x_dim.size() * 2, int64_t(paddings.size()),
                      "Size of paddings should be equal to 2 * dimension size "
                      "of input tensor.");
    for (size_t i = 0; i < paddings.size(); ++i) {
      PADDLE_ENFORCE_GE(paddings[i], 0, "paddings should >= 0.");
    }
    std::vector<int64_t> out_dims(x_dim.size());
    for (int i = 0; i < x_dim.size(); ++i) {
      if ((!ctx->IsRuntime()) && (x_dim[i] == -1)) {
        out_dims[i] = -1;
      } else {
        out_dims[i] = x_dim[i] + paddings[i * 2] + paddings[i * 2 + 1];
      }
    }
    ctx->SetOutputDim("Out", framework::make_ddim(out_dims));
    if (out_dims[0] == x_dim[0]) {
      // Only pass LoD when the first dimension is equal between
      // output and input.
      ctx->ShareLoD("X", /*->*/ "Out");
    }
  }
};

class PadOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input of pad op. "
             "The input should be a k-D tensor(k > 0 and k < 7)");
    AddOutput("Out",
              "The output of pad op. "
              "A tensor with the same shape as X.");
    AddAttr<std::vector<int>>(
        "paddings",
        "(vector<int>) "
        "A list<int> to describe the padding rules for each dimension. "
        "For 2-D image tensor, paddings=[0, 1, 2, 3] means "
        "padding 0 row to top, 1 row to bottom, 2 columns to left "
        "and 3 columns to right. Size of paddings should be equal to "
        "2 * dimension size of the input tensor.");
    AddAttr<float>("pad_value",
                   "(float, default 0.0) "
                   "The value to fill the padded areas.")
        .SetDefault(0.0f);
    AddComment(R"DOC(
Pad Operator.

Pad input into output, as specified by paddings and pad_value. 
The input should be a k-D tensor(k > 0 and k < 7). As an example:

Given:

X = [[1, 2],
     [3, 4]],

paddings = [0, 1, 1, 2],

and

pad_value = 0,

we have:

Out = [[0, 1, 2, 0, 0]
       [0, 3, 4, 0, 0]
       [0, 0, 0, 0, 0]]

)DOC");
  }
};

class PadOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      auto dout_dims = ctx->GetInputDim(framework::GradVarName("Out"));
      auto& paddings = ctx->Attrs().Get<std::vector<int>>("paddings");
      for (int i = 0; i < dout_dims.size(); ++i) {
        if (ctx->IsRuntime() || (dout_dims[i] != -1)) {
          dout_dims[i] -= (paddings[i * 2] + paddings[i * 2 + 1]);
        }
      }
      ctx->SetOutputDim(x_grad_name, dout_dims);
    }
  }
};

template <typename T>
class PadOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  std::unique_ptr<T> Apply() const override {
    auto* bind = new T();
    bind->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    bind->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    bind->SetAttrMap(this->Attrs());
    bind->SetType("pad_grad");
    return std::unique_ptr<T>(bind);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(pad, ops::PadOp, ops::PadOpMaker,
                  ops::PadOpGradMaker<paddle::framework::OpDesc>,
                  ops::PadOpGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(pad_grad, ops::PadOpGrad);
REGISTER_OP_CPU_KERNEL(
    pad, ops::PadKernel<paddle::platform::CPUDeviceContext, float>,
    ops::PadKernel<paddle::platform::CPUDeviceContext, double>,
    ops::PadKernel<paddle::platform::CPUDeviceContext, int>,
    ops::PadKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    pad_grad, ops::PadGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::PadGradKernel<paddle::platform::CPUDeviceContext, double>);
