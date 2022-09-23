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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

class GatherNdOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::Tensor>("X");
    const auto& x_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(
        x_type,
        x_type == framework::proto::VarType::BOOL
            ? x->place()  // to be consistent with compare and logical ops
            : ctx.device_context().GetPlace());
  }
};

class GatherNdGradOp : public framework::OperatorWithKernel {
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

class GatherNdOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The source input of gather_nd op");
    AddInput("Index", "The index input of gather_nd op");
    AddOutput("Out", "The output of gather_nd op");
    AddComment(R"DOC(
    Gather_Nd Operator.

    This function is actually a high-dimensional extension of gather
    and supports for simultaneous indexing by multiple axes. Out is
    obtained by gathering slices from X into a tensor with shape
    Index.shape[:-1] + X.shape[Index.shape[-1]:].

    Example:

    Given:
         X = [[[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]],
              [[12, 13, 14, 15],
               [16, 17, 18, 19],
               [20, 21, 22, 23]]]

         X.shape = (2, 3, 4)

   *Case 1:

       Index = [[1]]

    we get:
       Out =
            [[12, 13, 14, 15],
             [16, 17, 18, 19],
             [20, 21, 22, 23]]

   *Case 2:

       Index = [[0,2]]

    we get:

       Out =  [8, 9, 10, 11]

   *Case 3:

       Index = [[1, 2, 3]]

    we get:

       Out = [23]

)DOC");
  }
};

template <typename T>
class GatherNdGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("gather_nd_grad");
    op->SetInput("Index", this->Input("Index"));
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(GatherNdGradNoNeedBufferVarInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(gather_nd,
                            GatherNdInferShapeFunctor,
                            PD_INFER_META(phi::GatherNdInferMeta));

DECLARE_INFER_SHAPE_FUNCTOR(gather_nd_grad,
                            GatherNdGradInferShapeFunctor,
                            PD_INFER_META(phi::GatherNdGradInferMeta));

REGISTER_OPERATOR(gather_nd,
                  ops::GatherNdOp,
                  ops::GatherNdOpMaker,
                  ops::GatherNdGradOpMaker<paddle::framework::OpDesc>,
                  ops::GatherNdGradOpMaker<paddle::imperative::OpBase>,
                  GatherNdInferShapeFunctor);

REGISTER_OPERATOR(gather_nd_grad,
                  ops::GatherNdGradOp,
                  ops::GatherNdGradNoNeedBufferVarInferer,
                  GatherNdGradInferShapeFunctor);
