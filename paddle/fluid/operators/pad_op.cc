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

#include <memory>
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class PadOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Pad");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Pad");
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
  void Apply(GradOpPtr<T> bind) const override {
    bind->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    bind->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    bind->SetAttrMap(this->Attrs());
    bind->SetType("pad_grad");
  }
};

template <typename T>
class PadOpDoubleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("pad");
    grad_op->SetInput("X", this->OutputGrad(framework::GradVarName("X")));
    grad_op->SetOutput("Out", this->InputGrad(framework::GradVarName("Out")));
    grad_op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(pad, PadInferShapeFunctor,
                            PD_INFER_META(phi::PadInferMeta));

REGISTER_OPERATOR(pad, ops::PadOp, ops::PadOpMaker,
                  ops::PadOpGradMaker<paddle::framework::OpDesc>,
                  ops::PadOpGradMaker<paddle::imperative::OpBase>,
                  PadInferShapeFunctor);
REGISTER_OPERATOR(pad_grad, ops::PadOpGrad,
                  ops::PadOpDoubleGradMaker<paddle::framework::OpDesc>,
                  ops::PadOpDoubleGradMaker<paddle::imperative::OpBase>);
