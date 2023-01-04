// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>

#include "paddle/fluid/operators/elementwise/elementwise_op.h"

namespace paddle {
namespace operators {

class ElementwiseHeavisideOpMaker : public ElementwiseOpMaker {
 protected:
  std::string GetName() const override { return "Heaviside"; }
  std::string GetEquation() const override { return "Out = Heaviside(X, Y)"; }

  void AddInputX() override {
    AddInput("X",
             "(Tensor), The input tensor of Heaviside step function. "
             "Its dtype can be int32, int64, float32 and float64");
  }

  void AddInputY() override {
    AddInput("Y",
             "(Tensor), The tensor determining a Heaviside step function, "
             "which is the value when X = 0. Its dtype should be same as X.");
  }

  std::string GetOpFuntionality() const override {
    return "Computes the Heaviside step function determined by Y "
           "for each element in X.";
  }
};

template <typename T>
class ElementwiseHeavisideGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("elementwise_heaviside_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    elementwise_heaviside,
    ops::ElementwiseOp,
    ops::ElementwiseHeavisideOpMaker,
    ops::ElementwiseHeavisideGradOpMaker<paddle::framework::OpDesc>,
    ops::ElementwiseHeavisideGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(elementwise_heaviside_grad, ops::ElementwiseOpGrad);
