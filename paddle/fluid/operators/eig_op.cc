// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/eig_op.h"

#include <string>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class EigOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  // The output of eig is always complex-valued even for real-valued inputs
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto dtype = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    if (dtype != framework::proto::VarType::FP32 &&
        dtype != framework::proto::VarType::FP64 &&
        dtype != framework::proto::VarType::COMPLEX64 &&
        dtype != framework::proto::VarType::COMPLEX128) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "unsupported data type: %s!", dtype));
    }
    return framework::OpKernelType(dtype, ctx.GetPlace());
  }
};

class EigOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "X",
        "(Tensor), A complex-valued or real-valued tensor with shape (*, "
        "n, n). The accepted datatype is one of float32, float64, complex64 "
        "or complex128");
    AddOutput("Eigenvalues",
              "(Tensor), The output eigenvalues tensor with shape (*, n). The "
              "datatype is complex64 or complex128");
    AddOutput("Eigenvectors",
              "(Tensor), The output eigenvectors tensor with shape (*, n, n). "
              "The datatype is complex64 or complex128");

    AddComment(R"DOC(
        Eig Operator.

This API processes eigen decomposition for general square matrices.

)DOC");
  }
};

class EigGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(
            ctx, framework::GradVarName("Eigenvectors")),
        ctx.device_context());
  }
};

template <typename T>
class EigGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("Eigenvalues", this->Output("Eigenvalues"));
    op->SetInput("Eigenvectors", this->Output("Eigenvectors"));
    op->SetInput(framework::GradVarName("Eigenvalues"),
                 this->OutputGrad("Eigenvalues"));
    op->SetInput(framework::GradVarName("Eigenvectors"),
                 this->OutputGrad("Eigenvectors"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(eig,
                            EigInferShapeFunctor,
                            PD_INFER_META(phi::EigInferMeta));

DECLARE_INFER_SHAPE_FUNCTOR(eig_grad,
                            EigGradInferShapeFunctor,
                            PD_INFER_META(phi::EigGradInferMeta));

REGISTER_OPERATOR(eig,
                  ops::EigOp,
                  ops::EigOpMaker,
                  ops::EigGradOpMaker<paddle::framework::OpDesc>,
                  ops::EigGradOpMaker<paddle::imperative::OpBase>,
                  EigInferShapeFunctor);

REGISTER_OPERATOR(eig_grad, ops::EigGradOp, EigGradInferShapeFunctor);
