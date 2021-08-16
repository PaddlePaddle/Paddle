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

#include <cstring>
#include <string>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class ClearFloatStatusOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasOutput("FloatStatusOut"), "Output", "FloatStatusOut",
                   "clear_float_status");
    ctx->SetOutputDim("FloatStatusOut", ctx->GetInputDim("FloatStatus"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(framework::proto::VarType::FP32,
                                   ctx.GetPlace());
  }
};

class ClearFloatStatusMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("FloatStatus",
             "(Tensor) of shape {8} that holds the float status.");
    AddOutput(
        "FloatStatusOut",
        "(Tensor) of shape {8} that holds the float status, which is cleared.");
    AddComment(R"DOC(
      Clear the float status
)DOC");
  }
};

template <typename DeviceContext, typename T>
class ClearFloatStatusKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Operator clear_float_status is not supported on CPU"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(
    clear_float_status, ops::ClearFloatStatusOp, ops::ClearFloatStatusMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(clear_float_status,
                       ops::ClearFloatStatusKernel<CPU, float>);
