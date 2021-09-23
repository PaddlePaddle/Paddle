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

#include "paddle/fluid/operators/collective/assign_pos_op.h"

namespace paddle {
namespace operators {

class AssignPosOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("cum_count"), "Input", "cum_count", "AssignPos");
    OP_INOUT_CHECK(ctx->HasInput("eff_gates_len"), "Input", "eff_gates_len", "AssignPos");
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "AssignPos");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "AssignPos");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "cum_count");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class AssignPosOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The tensor which indicates the tokens belong to which topk experts.");
    AddInput("cum_count",
             "The cumulative sum tokens of experts.");
    AddInput("eff_gates_len",
             "The effective numbers of tokens should be sent.");         
    AddOutput("Out", "Assemble tokens in the order of experts.");

    AddComment(R"DOC(
assign_pos_op Operator.

Assign pos decides which tokens should be fetched belong to 
specially expert orderingly.


)DOC");
  }
};



}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_WITHOUT_GRADIENT(assign_pos, ops::AssignPosOp,
                             ops::AssignPosOpMaker);


// REGISTER_OPERATOR(assign_pos, ops::AssignPosOp, ops::AssignPosOpMaker)

REGISTER_OP_CPU_KERNEL(assign_pos, 
                       ops::AssignPosOpCPUKernel<int>,
                       ops::AssignPosOpCPUKernel<int64_t>);

