// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"

#if defined(PADDLE_WITH_GLOO)
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#endif

namespace paddle {
namespace operators {

class LimitByCapacityOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("expert_count"),
                   "Input",
                   "expert_count",
                   "LimitByCapacity");
    OP_INOUT_CHECK(
        ctx->HasInput("capacity"), "Input", "capacity", "LimitByCapacity");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "LimitByCapacity");

    ctx->ShareDim("expert_count", "Out");
    ctx->ShareLoD("expert_count", "Out");
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    // the dtype of the expert_count and capacity should be same as int64
    auto expert_count_dtype =
        OperatorWithKernel::IndicateVarDataType(ctx, "expert_count");
    auto capacity_dtype =
        OperatorWithKernel::IndicateVarDataType(ctx, "capacity");

    PADDLE_ENFORCE_EQ(
        expert_count_dtype,
        capacity_dtype,
        platform::errors::InvalidArgument(
            "The dtype of the expert_count and capacity should be same"));

    PADDLE_ENFORCE_EQ(
        expert_count_dtype,
        framework::proto::VarType::INT64,
        platform::errors::InvalidArgument("The dtype of the expert_count and "
                                          "capacity should be same as int64"));
    return phi::KernelKey(expert_count_dtype, ctx.GetPlace());
  }
};

class LimitByCapacityOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("expert_count", "(Tensor) The input expert count tensor.");
    AddInput("capacity", "(Tensor) The input capacity.");
    AddOutput("Out",
              "(Tensor) The output tensor expert count limit by capacity.");
    AddAttr<int>("n_worker", "（int), The number of works.");
    AddComment(
        R"DOC(limit_by_capacity Operator.limit expert count by capacity.)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(limit_by_capacity,
                             ops::LimitByCapacityOp,
                             ops::LimitByCapacityOpMaker);
