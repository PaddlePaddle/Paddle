/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include <unordered_map>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/matrix_inverse.h"

namespace paddle {
namespace operators {

class InverseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class InverseOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{
        {"Input", /*->*/ "Output"}};
    return m;
  }
};

class InverseGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class InverseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "Input",
        "(Tensor) A square matrix (2-D Tensor) or batches of square matrices"
        " to inverse.");
    AddOutput("Output", "(Tensor) The inverse of input matrix.");
    AddComment(R"DOC(
Inverse Operator

Takes the inverse of the square matrix.
)DOC");
  }
};

template <typename T>
class InverseGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad) const override {
    grad->SetType(this->ForwardOpType() + "_grad");
    grad->SetInput("Output", this->Output("Output"));
    grad->SetInput(framework::GradVarName("Output"),
                   this->OutputGrad("Output"));
    grad->SetOutput(framework::GradVarName("Input"), this->InputGrad("Input"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(inverse,
                            InverseInferShapeFunctor,
                            PD_INFER_META(phi::InverseInferMeta));

DECLARE_INFER_SHAPE_FUNCTOR(inverse_grad,
                            InverseGradInferShapeFunctor,
                            PD_INFER_META(phi::InverseGradInferMeta));

REGISTER_OPERATOR(inverse,
                  ops::InverseOp,
                  ops::InverseOpMaker,
                  ops::InverseOpInferVarType,
                  ops::InverseGradOpMaker<paddle::framework::OpDesc>,
                  ops::InverseGradOpMaker<paddle::imperative::OpBase>,
                  InverseInferShapeFunctor);

REGISTER_OPERATOR(inverse_grad,
                  ops::InverseGradOp,
                  InverseGradInferShapeFunctor);
