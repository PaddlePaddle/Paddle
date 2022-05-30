/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/assign_pos_op.h"

namespace paddle {
namespace operators {

class AssignPosOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("cum_count"), "Input", "cum_count",
                   "AssignPos");
    OP_INOUT_CHECK(ctx->HasInput("eff_num_len"), "Input", "eff_num_len",
                   "AssignPos");
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "AssignPos");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "AssignPos");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto cum_count_dtype =
        OperatorWithKernel::IndicateVarDataType(ctx, "cum_count");
    auto X_dtype = OperatorWithKernel::IndicateVarDataType(ctx, "X");

    PADDLE_ENFORCE_EQ(cum_count_dtype, X_dtype,
                      platform::errors::InvalidArgument(
                          "The dtype of the cum_count and X should be same"));
    PADDLE_ENFORCE_EQ(cum_count_dtype, framework::proto::VarType::INT64,
                      platform::errors::InvalidArgument(
                          "The dtype of the cum_count_dtype, eff_num_len and "
                          "X should be same as int64"));
    return framework::OpKernelType(cum_count_dtype, ctx.device_context());
  }
};

class AssignPosOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "numbers to scatter.");
    AddInput("cum_count", "The cumulative sum count of numbers.");
    AddInput("eff_num_len",
             "The effective numbers of numbers should be scattered.");
    AddOutput("Out", "Assemble numbers in the order of counters.");

    AddComment(R"DOC(
assign_pos_op Operator.

Assign pos decides which tokens should be fetched belong to 
specially counter orderingly.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_WITHOUT_GRADIENT(assign_pos, ops::AssignPosOp,
                             ops::AssignPosOpMaker);

REGISTER_OP_CPU_KERNEL(assign_pos, ops::AssignPosOpCPUKernel<int>,
                       ops::AssignPosOpCPUKernel<int64_t>);
