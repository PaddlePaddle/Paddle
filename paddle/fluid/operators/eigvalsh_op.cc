/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class EigvalshOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class EigvalshOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor), Hermitian or real symmetric matrices."
             "Its shape should be [*, N, N] where * is zero or"
             "more batch dimensions. The data type is float32 ,"
             "float64, complex64, complex128.");
    AddOutput("Eigenvalues",
              "(Tensor), The eigenvalues in ascending order."
              "The data type is float32 or float64.");
    AddOutput(
        "Eigenvectors",
        "(Tensor), The column is the normalized eigenvector "
        "corresponding to the eigenvalue. The data type is the same as ``X``."
        "Eigenvectors are required to calculate gradient when backward.");
    AddAttr<std::string>(
        "UPLO",
        "(string, default 'L'), 'L' represents the lower triangular matrix,"
        "'U' represents the upper triangular matrix.")
        .SetDefault("L");
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training.")
        .SetDefault(false);
    AddComment(R"DOC(
Eigvalsh Operator.

Computes the eigenvalues of a complex Hermitian
 (conjugate symmetric) or a real symmetric matrix.

)DOC");
  }
};

class EigvalshGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Eigenvectors"),
        ctx.device_context());
  }
};

template <typename T>
class EigvalshGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("Eigenvectors", this->Output("Eigenvectors"));
    op->SetInput(framework::GradVarName("Eigenvalues"),
                 this->OutputGrad("Eigenvalues"));
    op->SetAttrMap(this->Attrs());
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(eigvalsh,
                            EigvalshInferShapeFunctor,
                            PD_INFER_META(phi::EigvalshInferMeta));
DECLARE_INFER_SHAPE_FUNCTOR(eigvalsh_grad,
                            EigvalshGradInferShapeFunctor,
                            PD_INFER_META(phi::EigvalshGradInferMeta));

REGISTER_OPERATOR(eigvalsh,
                  ops::EigvalshOp,
                  ops::EigvalshOpMaker,
                  ops::EigvalshGradOpMaker<paddle::framework::OpDesc>,
                  ops::EigvalshGradOpMaker<paddle::imperative::OpBase>,
                  EigvalshInferShapeFunctor);
REGISTER_OPERATOR(eigvalsh_grad,
                  ops::EigvalshGradOp,
                  EigvalshGradInferShapeFunctor);
