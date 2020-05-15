// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/allclose_op.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {

class AllcloseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "The first input tensor to compare.");
    AddInput("Other", "The second input tensor to compare.");
    AddOutput("Out", "The output tensor of allclose op.");

    AddAttr<float>("rtol", "The relative tolerance. Default: :math:`1e-5` .")
        .SetDefault(1e-5);
    AddAttr<float>("atol", "The absolute tolerance. Default: :math:`1e-8` .")
        .SetDefault(1e-8);
    AddAttr<bool>("equal_nan",
                  "If :math:`True` , then two :math:`NaNs` will be "
                  "compared as equal. Default: :math:`False` .")
        .SetDefault(false);

    AddComment(R"DOC( 
This operator checks if all :math:`input` and :math:`other` satisfy the condition:

:math:`\left| input - other \right| \leq atol + rtol \times \left| other \right|`

elementwise, for all elements of :math:`input` and :math:`other`. The behaviour of this
operator is analogous to :math:`numpy.allclose`, namely that it returns :math:`True` if
two tensors are elementwise equal within a tolerance.
)DOC");
  }
};

class AllcloseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("Input"), true,
                      platform::errors::NotFound(
                          "Input(Input) of allclose op should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Other"), true,
                      platform::errors::NotFound(
                          "Input(Other) of allclose op should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::NotFound(
                          "The output(Out) of allclose op must not be null."));

    auto input_dim = ctx->GetInputDim("Input");
    auto other_dim = ctx->GetInputDim("Other");
    PADDLE_ENFORCE_EQ(input_dim.size(), other_dim.size(),
                      platform::errors::PreconditionNotMet(
                          "Input(Input) and Input(Other) must have the same "
                          "dimension size."));
    int n = input_dim.size();
    bool is_runtime = ctx->IsRuntime();
    for (int i = 0; i < n; i++) {
      if (is_runtime) {
        PADDLE_ENFORCE_EQ(input_dim[i], other_dim[i],
                          platform::errors::PreconditionNotMet(
                              "The value at dim %d of Input(Input) is not "
                              "equal to the Input(Other): %ld != %ld.",
                              i, input_dim[i], other_dim[i]));
      } else {
        if (!(input_dim[i] < 0 || other_dim[i] < 0)) {
          PADDLE_ENFORCE_EQ(input_dim[i], other_dim[i],
                            platform::errors::PreconditionNotMet(
                                "The value at dim %d of Input(Input) is not "
                                "equal to the Input(Other): %ld != %ld.",
                                i, input_dim[i], other_dim[i]));
        }
      }
    }

    ctx->SetOutputDim("Out", framework::make_ddim({1}));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
        ctx.device_context());
  }
};

class AllcloseOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    ctx->SetOutputDataType("Out", framework::proto::VarType::BOOL);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(
    allclose, ops::AllcloseOp, ops::AllcloseOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    ops::AllcloseOpVarTypeInference);
REGISTER_OP_CPU_KERNEL(allclose, ops::AllcloseKernel<CPU, float>,
                       ops::AllcloseKernel<CPU, double>);
