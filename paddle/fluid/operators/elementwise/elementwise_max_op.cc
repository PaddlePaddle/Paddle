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

#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/prim/api/composite_backward/composite_backward_api.h"
#include "paddle/fluid/prim/utils/static/composite_grad_desc_maker.h"
#include "paddle/fluid/prim/utils/static/desc_tensor.h"

namespace paddle {
namespace framework {
class OpDesc;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {

class ElementwiseMaxOpMaker : public ElementwiseOpMaker {
 protected:
  std::string GetName() const override { return "Max"; }
  std::string GetEquation() const override { return "Out = max(X, Y)"; }

  void AddInputX() override {
    AddInput("X", "The first tensor holding the elements to be compared.");
  }

  void AddInputY() override {
    AddInput("Y", "The second tensor holding the elements to be compared.");
  }

  std::string GetOpFuntionality() const override {
    return "Compare two tensors and returns a new tensor containing the "
           "element-wise maxima.";
  }
};

class ElementwiseFMaxOpMaker : public ElementwiseOpMaker {
 protected:
  std::string GetName() const override { return "FMax"; }
  std::string GetEquation() const override { return "Out = fmax(X, Y)"; }

  void AddInputX() override {
    AddInput("X", "The first tensor holding the elements to be compared.");
  }

  void AddInputY() override {
    AddInput("Y", "The second tensor holding the elements to be compared.");
  }

  std::string GetOpFuntionality() const override {
    return "Compare two tensors and returns a new tensor containing the "
           "element-wise maxima. If the element of one tensor is nan, "
           "return the element value of the other tensor, if both are nan, "
           "return the first nan";
  }
};

class ElementwiseMaxCompositeGradOpMaker
    : public prim::CompositeGradOpMakerBase {
  using prim::CompositeGradOpMakerBase::CompositeGradOpMakerBase;

 public:
  void Apply() override {
    paddle::Tensor x = this->GetSingleForwardInput("X");
    paddle::Tensor y = this->GetSingleForwardInput("Y");
    paddle::Tensor out_grad = this->GetSingleOutputGrad("Out");
    paddle::Tensor dx = this->GetSingleInputGrad("X");
    auto* dx_ptr = this->GetOutputPtr(&dx);
    std::string dx_name = this->GetOutputName(dx);
    paddle::Tensor dy = this->GetSingleInputGrad("Y");
    auto* dy_ptr = this->GetOutputPtr(&dy);
    std::string dy_name = this->GetOutputName(dy);
    VLOG(6) << "Runing maximum_grad composite func";
    prim::maximum_grad<prim::DescTensor>(x, y, out_grad, dx_ptr, dy_ptr);
    this->RecoverOutputName(dx, dx_name);
    this->RecoverOutputName(dy, dy_name);
  }
};

template <typename T>
class ElementwiseMaxGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("elementwise_max_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    op->SetAttrMap(this->Attrs());
  }
};

template <typename T>
class ElementwiseFMaxGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("elementwise_fmax_grad");
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

REGISTER_OPERATOR(elementwise_max,
                  ops::ElementwiseOp,
                  ops::ElementwiseMaxOpMaker,
                  ops::ElementwiseOpInferVarType,
                  ops::ElementwiseMaxGradOpMaker<paddle::framework::OpDesc>,
                  ops::ElementwiseMaxGradOpMaker<paddle::imperative::OpBase>,
                  ops::ElementwiseMaxCompositeGradOpMaker);

REGISTER_OPERATOR(elementwise_max_grad, ops::ElementwiseOpGrad);

REGISTER_OP_VERSION(elementwise_max)
    .AddCheckpoint(
        R"ROC(Register elementwise_max for adding the attribute of Scale_y)ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "Scale_y",
            "In order to support the function of scaling the input Y when "
            "using the operator of elementwise_max.",
            1.0f));

REGISTER_OPERATOR(elementwise_fmax,
                  ops::ElementwiseOp,
                  ops::ElementwiseFMaxOpMaker,
                  ops::ElementwiseOpInferVarType,
                  ops::ElementwiseFMaxGradOpMaker<paddle::framework::OpDesc>,
                  ops::ElementwiseFMaxGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(elementwise_fmax_grad, ops::ElementwiseOpGrad);
