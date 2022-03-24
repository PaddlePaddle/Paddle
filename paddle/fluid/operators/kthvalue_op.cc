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

#include <memory>
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class KthvalueOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class KthvalueOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddComment(R"DOC(
    This operator find the k-th smallest elements in the specific axis of a Tensor.
    It will return the values and corresponding indices.
    )DOC");
    AddInput("X", "(Tensor) The input of Kthvalue op");
    AddOutput("Out", "(Tensor) The values of k-th smallest elements of input");
    AddOutput("Indices",
              "(Tensor) The indices of k-th smallest elements of input");
    AddAttr<int>(
        "k",
        "(int, default 1) k for k-th smallest elements to look for along "
        "the tensor).")
        .SetDefault(1);
    AddAttr<int>("axis",
                 "the axis to sort and get the k indices, value."
                 "if not set, will get k-th value in last axis.")
        .SetDefault(-1);
    AddAttr<bool>("keepdim", "Keep the dim that to reduce.").SetDefault(false);
  }
};

class KthvalueOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::InvalidArgument("Input(X) should be not null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Indices"), true,
        platform::errors::InvalidArgument("Input(Indices) should be not null"));
    PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Out")), true,
                      platform::errors::InvalidArgument(
                          "Grad Input(Out) should be not null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput(framework::GradVarName("X")), true,
        platform::errors::InvalidArgument("Grad Output(X) should be not null"));

    auto x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

template <typename T>
class KthvalueGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("kthvalue_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput("X", this->Input("X"));
    op->SetInput("Indices", this->Output("Indices"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

DECLARE_INFER_SHAPE_FUNCTOR(kthvalue, KthvalueInferShapeFunctor,
                            PD_INFER_META(phi::KthvalueInferMeta));

namespace ops = paddle::operators;
REGISTER_OPERATOR(kthvalue, ops::KthvalueOp, ops::KthvalueOpMaker,
                  ops::KthvalueGradOpMaker<paddle::framework::OpDesc>,
                  ops::KthvalueGradOpMaker<paddle::imperative::OpBase>,
                  KthvalueInferShapeFunctor);

REGISTER_OPERATOR(kthvalue_grad, ops::KthvalueOpGrad);
