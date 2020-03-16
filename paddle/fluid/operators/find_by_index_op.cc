/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/find_by_index_op.h"
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/sampler.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
class FindByIndexOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("Input", "Input(Tensor), dtype support int64");
    AddInput("Index", "Index(Tensor), dtype support int64");
    AddOutput("Out", "Return the  element of index");

    AddComment(R"DOC("FindByIndex")DOC");
  }
};

class FindByIndexOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("Input"), true,
                      "Inputs(Input) of FindByIndex should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("Index"), true,
                      "Inputs(Index) of FindByIndex should not be null.");

    auto index_dims = ctx->GetInputDim("Index");
    ctx->SetOutputDim("Out", index_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "Input");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    find_by_index, ops::FindByIndexOp, ops::FindByIndexOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    find_by_index, ops::FindByIndexKernel<paddle::platform::CPUPlace, float>,
    ops::FindByIndexKernel<paddle::platform::CPUPlace, double>,
    ops::FindByIndexKernel<paddle::platform::CPUPlace, int>,
    ops::FindByIndexKernel<paddle::platform::CPUPlace, int64_t>);
