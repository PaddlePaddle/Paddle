/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License. */

#include "paddle/fluid/operators/bmm_op.h"

#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

class BmmOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class BmmOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The first input tensor of Bmm op.");
    AddInput("Y", "(Tensor), The second input tensor of Bmm op.");
    AddOutput("Out", "(Tensor), The output tensor of Bmm op.");
    AddComment(R"DOC(
The Bmm operator is used to perform batched matrix multiplication
over the last two dimensions of the input tensors `X` and `Y`
which are both 3-dimentionsal.

Examples:
- X: [B, M, K], Y: [B, K, N] => Out: [B, M, N]

      )DOC");
  }
};

class BmmOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

template <typename T>
class BmmOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("bmm_grad");
    retv->SetInput("X", this->Input("X"));
    retv->SetInput("Y", this->Input("Y"));
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(bmm,
                            BmmInferShapeFunctor,
                            PD_INFER_META(phi::BmmInferMeta));
DECLARE_INFER_SHAPE_FUNCTOR(bmm_grad,
                            BmmGradInferShapeFunctor,
                            PD_INFER_META(phi::BmmGradInferMeta));
REGISTER_OPERATOR(bmm,
                  ops::BmmOp,
                  ops::BmmOpMaker,
                  ops::BmmOpGradMaker<paddle::framework::OpDesc>,
                  ops::BmmOpGradMaker<paddle::imperative::OpBase>,
                  BmmInferShapeFunctor);
REGISTER_OPERATOR(bmm_grad, ops::BmmOpGrad, BmmGradInferShapeFunctor);
