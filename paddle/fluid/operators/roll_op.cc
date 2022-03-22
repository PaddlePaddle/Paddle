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

#include <memory>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class RollOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class RollGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Out")), true,
                      platform::errors::InvalidArgument(
                          "Input(Out@GRAD) should be not null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput(framework::GradVarName("X")), true,
                      platform::errors::InvalidArgument(
                          "Output(X@GRAD) should be not null."));

    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

class RollOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) the input tensor.");
    AddOutput("Out", "(Tensor), the output tensor.");
    AddAttr<std::vector<int64_t>>("shifts",
                                  "The number of places by which the elements "
                                  "of the tensor are shifted.")
        .SetDefault({});
    AddInput("ShiftsTensor",
             "The number of places by which the elements of the tensor "
             "are shifted.")
        .AsDispensable();
    AddAttr<std::vector<int64_t>>(
        "axis",
        "Axis along which to roll. It must have the same size "
        "with shifts or size == 0")
        .SetDefault({});
    AddComment(R"DOC(
    Roll the tensor along the given dimension(s). 
    Elements that are shifted beyond the last position
    are re-introduced at the first position. If a dimension
    is not specified, the tensor will be flattened before 
    rolling and then restored to the original shape.
    )DOC");
  }
};

template <typename T>
class RollGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("roll_grad");
    op->SetInput("X", this->Input("X"));
    if (this->HasInput("ShiftsTensor")) {
      op->SetInput("ShiftsTensor", this->Input("ShiftsTensor"));
    }
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(RollGradNoNeedBufferVarsInferer, "X");
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(roll, RollInferShapeFunctor,
                            PD_INFER_META(phi::RollInferMeta));

REGISTER_OPERATOR(roll, ops::RollOp, ops::RollOpMaker,
                  ops::RollGradMaker<paddle::framework::OpDesc>,
                  ops::RollGradMaker<paddle::imperative::OpBase>,
                  RollInferShapeFunctor);
REGISTER_OPERATOR(roll_grad, ops::RollGradOp,
                  ops::RollGradNoNeedBufferVarsInferer);

REGISTER_OP_VERSION(roll)
    .AddCheckpoint(
        R"ROC(
      Upgrade roll add 1 attribute [axis], delete 1 attribute[dims].
    )ROC",
        paddle::framework::compatible::OpVersionDesc()
            .NewAttr("axis",
                     "(std::vector<int64_t>) Axis along which to roll. "
                     "It must have the same size with shifts, or size = 0.",
                     std::vector<int64_t>())
            .DeleteAttr("dims",
                        "(std::vector<int64_t>) Dims along which to roll. "
                        "It must have the same size with shifts, or size = 0."))
    .AddCheckpoint(
        R"ROC(Upgrade roll add a dispensable input "ShiftsTensor".)ROC",
        paddle::framework::compatible::OpVersionDesc().NewInput(
            "ShiftsTensor",
            "The number of places by which the elements of"
            "the tensor are shifted."));
