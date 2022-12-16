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
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class ExponentialOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class ExponentialOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddComment(R"DOC(
This operator fills the input tensor with random values sampled from a
exponential distribution.
)DOC");
    AddInput("X", "The input tensor.");
    AddOutput("Out", "The output tensor of exponential OP.");
    AddAttr<float>(
        "lambda", "lambd parameter of exponential distribution. [default 1.0].")
        .SetDefault(1.0f);
  }
};

template <typename T>
class ExponentialGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("fill_any_like");
    retv->SetInput("X", this->OutputGrad("Out"));
    retv->SetAttr("value", 0.0f);
    retv->SetOutput("Out", this->InputGrad("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

DECLARE_INPLACE_OP_INFERER(ExponentialInferer, {"X", "Out"});

DECLARE_INFER_SHAPE_FUNCTOR(exponential,
                            ExponentialInfershapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));

REGISTER_OPERATOR(exponential,
                  ops::ExponentialOp,
                  ops::ExponentialOpMaker,
                  ops::ExponentialGradOpMaker<paddle::framework::OpDesc>,
                  ops::ExponentialGradOpMaker<paddle::imperative::OpBase>,
                  ExponentialInferer,
                  ExponentialInfershapeFunctor);
