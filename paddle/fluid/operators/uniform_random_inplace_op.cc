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

class UniformRandomInplaceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddComment(R"DOC(
This operator fills self tensor with random values sampled from a
uniform distribution. The random result is in a range of [min, max).
)DOC");
    AddInput("X", "The input tensor.");
    AddOutput("Out", "The output tensor of uniform random op");
    AddAttr<float>("min", "Minimum value of uniform random. [default -1.0].")
        .SetDefault(-1.0f);
    AddAttr<float>("max", "Maximun value of uniform random. [default 1.0].")
        .SetDefault(1.0f);
    AddAttr<int>("seed",
                 "Random seed used for generating samples. "
                 "If seed is 0, it will use the seed of the global default "
                 "generator (which can be set by paddle.seed). "
                 "Note that if seed is not 0, this operator will always "
                 "generate the same random numbers every time. [default 0].")
        .SetDefault(0);
    AddAttr<int>("diag_num",
                 "The number of diag elements. Note that if "
                 "diag_num is 0, it means without diag init.[default 0].")
        .SetDefault(0);
    AddAttr<int>("diag_step", "The step between two diag element.[default 0].")
        .SetDefault(0);
    AddAttr<float>("diag_val", "The value of diag element. [default 1.0].")
        .SetDefault(1.0f);
  }
};

class UniformRandomInplaceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class UniformRandomInplaceGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class UniformRandomInplaceOpVarTypeInference
    : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {}
};

template <typename T>
class UniformRandomInplaceGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType(this->ForwardOpType() + "_grad");
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle
DECLARE_INPLACE_OP_INFERER(UniformRandomInplaceInferer, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(UniformRandomInplaceGradInplaceInferer,
                           {paddle::framework::GradVarName("Out"),
                            paddle::framework::GradVarName("X")});

DECLARE_INFER_SHAPE_FUNCTOR(uniform_random_inplace,
                            UniformRandomInplaceInferShapeFunctor,
                            PD_INFER_META(phi::UniformRandomInplaceInferMeta));
DECLARE_INFER_SHAPE_FUNCTOR(
    uniform_random_inplace_grad,
    UniformRandomInplaceGradInferShapeFunctor,
    PD_INFER_META(phi::UniformRandomInplaceGradInferMeta));

REGISTER_OPERATOR(uniform_random_inplace,
                  paddle::operators::UniformRandomInplaceOp,
                  paddle::operators::UniformRandomInplaceOpMaker,
                  paddle::operators::UniformRandomInplaceGradOpMaker<
                      paddle::framework::OpDesc>,
                  paddle::operators::UniformRandomInplaceGradOpMaker<
                      paddle::imperative::OpBase>,
                  paddle::operators::UniformRandomInplaceOpVarTypeInference,
                  UniformRandomInplaceInferer,
                  UniformRandomInplaceInferShapeFunctor);
REGISTER_OPERATOR(uniform_random_inplace_grad,
                  paddle::operators::UniformRandomInplaceGradOp,
                  UniformRandomInplaceGradInplaceInferer,
                  UniformRandomInplaceGradInferShapeFunctor);
