// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class TraceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class TraceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input",
             "(Tensor) The input tensor, from which the diagonals are taken.");
    AddOutput("Out", "(Tensor) the sum along diagonals of the input tensor");
    AddAttr<int>(
        "offset",
        R"DOC((int, default 0), offset of the diagonal from the main diagonal. Can be both positive and negative. Defaults to 0.
        )DOC")
        .SetDefault(0);
    AddAttr<int>(
        "axis1",
        R"DOC((int, default 0), the first axis of the 2-D planes from which the diagonals should be taken. 
        Can be either positive or negative. Default: 0.
        )DOC")
        .SetDefault(0);
    AddAttr<int>(
        "axis2",
        R"DOC((int, default 1), the second axis of the 2-D planes from which the diagonals should be taken. 
        Can be either positive or negative. Default: 1.
        )DOC")
        .SetDefault(1);
    AddComment(R"DOC(
Trace Operator.
Return the sum along diagonals of the input tensor.
The behavior of this operator is similar to how `numpy.trace` works.

If Input is 2-D, returns the sum of diagonal. 
If Input has larger dimensions, then returns an tensor of diagonals sum, diagonals be taken from
the 2-D planes specified by dim1 and dim2.

)DOC");
  }
};
class TraceGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Input"), true,
        platform::errors::NotFound("Input(Input) of TraceOp is not found."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput(framework::GradVarName("Input")), true,
                      platform::errors::NotFound(
                          "Output(Input@GRAD) of TraceGradOp is not found."));
    ctx->SetOutputDim(framework::GradVarName("Input"),
                      ctx->GetInputDim("Input"));
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
class TraceGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("trace_grad");
    grad_op->SetInput("Input", this->Input("Input"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("Input"),
                       this->InputGrad("Input"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(TraceGradNoNeedBufferVarsInferer, "Input");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(trace, TraceInferShapeFunctor,
                            PD_INFER_META(phi::TraceInferMeta));
REGISTER_OPERATOR(trace, ops::TraceOp, ops::TraceOpMaker,
                  ops::TraceGradOpMaker<paddle::framework::OpDesc>,
                  ops::TraceGradOpMaker<paddle::imperative::OpBase>,
                  TraceInferShapeFunctor);

REGISTER_OPERATOR(trace_grad, ops::TraceGradOp,
                  ops::TraceGradNoNeedBufferVarsInferer);

/* ==========================  register checkpoint ===========================*/
REGISTER_OP_VERSION(trace)
    .AddCheckpoint(
        R"ROC(Upgrade trace add a new attribute [axis2])ROC",
        paddle::framework::compatible::OpVersionDesc()
            .NewAttr("axis1",
                     "The added attribute 'axis1' is not yet registered.",
                     std::vector<float>{0.0f})
            .NewAttr("axis2",
                     "The added attribute 'axis2' is not yet registered.",
                     std::vector<float>{1.0f})
            .DeleteAttr("dim1",
                        "The attribute 'dim1' is not recommend according to "
                        "the specification 2.0.")
            .DeleteAttr("dim2",
                        "The attribute 'dim2' is not recommend according to "
                        "the specification 2.0."));
