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

#include "paddle/fluid/operators/expand_as_v2_op.h"
#include <memory>
#include <vector>
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class ExpandAsV2Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class ExpandAsV2OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor, default Tensor<float>). A tensor with rank in [1, 6]."
             "X is the input to be expanded.");
    AddInput("Y",
             "(Tensor, default Tensor<float>). A tensor with rank in [1, 6]."
             "Expand X according to the shape of Y.")
        .AsDispensable();
    AddOutput("Out",
              "(Tensor, default Tensor<float>). A tensor with rank in [1, 6]."
              "The rank of Output(Out) have the same with Input(X). "
              "After expanding, size of each dimension of Output(Out) is equal "
              "to size of the corresponding dimension of Input(X) multiplying "
              "the corresponding value given by Attr(expand_times).");
    AddAttr<std::vector<int>>("target_shape",
                              "Expand shape for each dimension.")
        .SetDefault({});
    AddComment(R"DOC(
Expand the input to the given shape.
)DOC");
  }
};

class ExpandAsV2GradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ExpandAsV2Grad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "ExpandAsV2Grad");

    auto x_dims = ctx->GetInputDim("X");
    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

template <typename T>
class ExpandAsV2GradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("expand_as_v2_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(ExpandAsV2GradNoNeedBufVarsInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(expand_as_v2, ExpandAsInferShapeFunctor,
                            PD_INFER_META(phi::ExpandAsInferMeta));
REGISTER_OPERATOR(expand_as_v2, ops::ExpandAsV2Op, ops::ExpandAsV2OpMaker,
                  ops::ExpandAsV2GradOpMaker<paddle::framework::OpDesc>,
                  ops::ExpandAsV2GradOpMaker<paddle::imperative::OpBase>,
                  ExpandAsInferShapeFunctor);
REGISTER_OPERATOR(expand_as_v2_grad, ops::ExpandAsV2GradOp,
                  ops::ExpandAsV2GradNoNeedBufVarsInferer);

REGISTER_OP_VERSION(expand_as_v2)
    .AddCheckpoint(
        R"ROC(fix expand_as_v2 and add new input [Y])ROC",
        paddle::framework::compatible::OpVersionDesc().NewInput(
            "Y", "Expand X according to the shape of Y"));
