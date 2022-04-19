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

#include <memory>
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/ternary.h"

namespace paddle {
namespace operators {

class ScatterOp : public framework::OperatorWithKernel {
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

class ScatterGradOp : public framework::OperatorWithKernel {
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

class ScatterOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The source input of scatter op");
    AddInput("Ids", "The index input of scatter op where X will be updated");
    AddInput("Updates", "The updated value of scatter op");
    AddOutput("Out", "The output of scatter op");
    AddAttr<bool>("overwrite",
                  "(bool, default: True) "
                  "The mode that updating the output when has same index,"
                  "If True, use the overwrite mode to update the output"
                  "of the same index, if False, use the accumulate mode to"
                  "update the output of the same index,Default value is True."
                  "You can set overwrite=False to implement scatter_add.")
        .SetDefault(true);
    AddComment(R"DOC(
Scatter Operator.

This operator obtains output by updating the input on selected indices on the first axis:

$$
Out = X \\
Out[Ids] = Updates
$$

)DOC");
  }
};

template <typename T>
class ScatterGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("scatter_grad");
    op->SetInput("Ids", this->Input("Ids"));
    op->SetInput("Updates", this->Input("Updates"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Updates"),
                  this->InputGrad("Updates"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(ScatterGradNoNeedBufferVarsInferer,
                                    "Updates");

DECLARE_INPLACE_OP_INFERER(ScatterInplaceInferer, {"X", "Out"});

}  // namespace operators
}  // namespace paddle

DECLARE_INFER_SHAPE_FUNCTOR(scatter, ScatterInferShapeFunctor,
                            PD_INFER_META(phi::ScatterInferMeta));

DECLARE_INFER_SHAPE_FUNCTOR(scatter_grad, ScatterGradInferShapeFunctor,
                            PD_INFER_META(phi::ScatterGradInferMeta));

namespace ops = paddle::operators;
REGISTER_OPERATOR(scatter, ops::ScatterOp, ops::ScatterOpMaker,
                  ops::ScatterGradMaker<paddle::framework::OpDesc>,
                  ops::ScatterGradMaker<paddle::imperative::OpBase>,
                  ops::ScatterInplaceInferer, ScatterInferShapeFunctor);
REGISTER_OPERATOR(scatter_grad, ops::ScatterGradOp,
                  ops::ScatterGradNoNeedBufferVarsInferer,
                  ScatterGradInferShapeFunctor);
