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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

class DiagV2Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class DiagV2OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor. Its shape is either 1-D or 2-D.");
    AddOutput("Out", "The output tensor. A square matrix or a vector.");
    AddAttr<int>("offset",
                 "The diagonal offset. A positive value represents "
                 "superdiagonal, 0 represents the main diagonal, and a "
                 "negative value represents subdiagonal.")
        .SetDefault(0);
    AddAttr<float>("padding_value",
                   "Use this value to fill the area outside the specified "
                   "diagonal band. Only takes effect when the input is a 1-D "
                   "Tensor. The default value is 0.")
        .SetDefault(0.0f);
    AddComment(R"DOC(
      If ``x`` is a vector (1-D tensor), a 2-D square tensor with the elements of ``x`` as the diagonal is returned.

      If ``x`` is a matrix (2-D tensor), a 1-D tensor with the diagonal elements of ``x`` is returned.

      The argument ``offset`` controls the diagonal offset:

      If ``offset`` = 0, it is the main diagonal.

      If ``offset`` > 0, it is superdiagonal.

      If ``offset`` < 0, it is subdiagonal.
)DOC");
  }
};

class DiagV2GradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "X", "X", "DiagV2Grad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Output",
                   framework::GradVarName("X"), "DiagV2Grad");

    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace());
  }
};

template <typename T>
class DiagV2GradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("diag_v2_grad");
    grad_op->SetInput("X", this->Input("X"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(DiagGradV2NoNeedBufferVarsInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(diag_v2, DiagInferShapeFunctor,
                            PD_INFER_META(phi::DiagInferMeta));

REGISTER_OPERATOR(diag_v2, ops::DiagV2Op, ops::DiagV2OpMaker,
                  ops::DiagV2GradOpMaker<paddle::framework::OpDesc>,
                  ops::DiagV2GradOpMaker<paddle::imperative::OpBase>,
                  DiagInferShapeFunctor);

REGISTER_OPERATOR(diag_v2_grad, ops::DiagV2GradOp,
                  ops::DiagGradV2NoNeedBufferVarsInferer);
