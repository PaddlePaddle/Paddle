/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include <vector>
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/ternary.h"

namespace paddle {
namespace operators {

class ScatterNdAddOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                      OperatorWithKernel::IndicateVarDataType(ctx, "Updates"),
                      platform::errors::InvalidArgument(
                          "Ref and Updates must have same type"));
    return framework::OpKernelType(
        framework::TransToProtoVarType(
            ctx.Input<framework::Tensor>("X")->type()),
        ctx.device_context());
  }
};

class ScatterNdAddGradOp : public framework::OperatorWithKernel {
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

class ScatterNdAddOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The source input of scatter_nd_add op");
    AddInput("Index",
             "The index input of scatter_nd_add op where X will be updated");
    AddInput("Updates", "The updated value of scatter_nd_add op");
    AddOutput("Out", "The output of scatter_nd_add op");
    AddComment(R"DOC(
Scatter_nd_add Operator.

Output is obtained by applying sparse addition to a single value or slice in a Variable.

      Given:
        * Case 1:
            ref = [0, 1, 2, 3, 4, 5]
            index = [[1], [2], [3], [1]]
            updates = [9, 10, 11, 12]

          we get:

            output = [0, 22, 12, 14, 4, 5]

        * Case 2:
            ref = [[65, 17], [-14, -25]]
            index = [[], []]
            updates = [[[-1, -2], [1, 2]],
                       [[3, 4], [-3, -4]]]
            ref.shape = (2, 2)
            index.shape = (2, 0)
            updates.shape = (2, 2, 2)

          we get:

            output = [[67, 19], [-16, -27]]
)DOC");
  }
};

template <typename T>
class ScatterNdAddGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("scatter_nd_add_grad");
    op->SetInput("Index", this->Input("Index"));
    op->SetInput("Updates", this->Input("Updates"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Updates"),
                  this->InputGrad("Updates"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(ScatterNdAddGradNoNeedBufferVarsInferer,
                                    "Updates");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(scatter_nd_add, ScatterNdAddInferShapeFunctor,
                            PD_INFER_META(phi::ScatterNdAddInferMeta));

DECLARE_INFER_SHAPE_FUNCTOR(scatter_nd_add_grad,
                            ScatterNdAddGradInferShapeFunctor,
                            PD_INFER_META(phi::ScatterNdAddGradInferMeta));

REGISTER_OPERATOR(scatter_nd_add, ops::ScatterNdAddOp, ops::ScatterNdAddOpMaker,
                  ops::ScatterNdAddGradMaker<paddle::framework::OpDesc>,
                  ops::ScatterNdAddGradMaker<paddle::imperative::OpBase>,
                  ScatterNdAddInferShapeFunctor);

REGISTER_OPERATOR(scatter_nd_add_grad, ops::ScatterNdAddGradOp,
                  ops::ScatterNdAddGradNoNeedBufferVarsInferer,
                  ScatterNdAddGradInferShapeFunctor);
