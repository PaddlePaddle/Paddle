// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/prune_gate_by_capacity_op.h"

namespace paddle {
namespace operators {

class PruneGateByCapacityOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("GateIdx"), "Input", "GateIdx",
                   "prun_gate_by_capacity");
    OP_INOUT_CHECK(ctx->HasInput("ExpertCount"), "Input", "ExpertCount",
                   "prun_gate_by_capacity");

    OP_INOUT_CHECK(ctx->HasOutput("NewGateIdx"), "Output", "NewGateIdx",
                   "prun_gate_by_capacity");
    // OP_INOUT_CHECK(ctx->HasOutput("ExpertCountOut"), "Output",
    // "ExpertCountOut",
    //                "prun_gate_by_capacity");
    // auto gate_idx_dims = ctx->GetInputDim("GateIdx");
    auto expert_count_dims = ctx->GetInputDim("ExpertCount");

    int64_t n_expert = ctx->Attrs().Get<int64_t>("n_expert");
    int64_t n_worker = ctx->Attrs().Get<int64_t>("n_worker");

    int64_t expert_count_num_ele = 1;
    for (int64_t i = 0; i < expert_count_dims.size(); i++) {
      expert_count_num_ele *= expert_count_dims[i];
    }

    PADDLE_ENFORCE_EQ(
        expert_count_num_ele, n_expert * n_worker,
        platform::errors::Unavailable(
            "The number of elements for expert_count is ( %ld ) incorrect. "
            "Because the number of expert_count must equal the "
            "product of n_worker ( %ld ) and n_expert ( %ld ). "
            "Please input appropriate expert_count again!",
            expert_count_num_ele, n_worker, n_expert));

    auto gate_idx_in_dims = ctx->GetInputDim("GateIdx");
    // auto expert_count_in_dims = ctx->GetInputDim("ExpertCount");
    ctx->SetOutputDim("NewGateIdx", gate_idx_in_dims);
    // ctx->SetOutputDim("ExpertCountOut", expert_count_in_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto gate_idx_data_type =
        OperatorWithKernel::IndicateVarDataType(ctx, "GateIdx");
    auto expert_count_data_type =
        OperatorWithKernel::IndicateVarDataType(ctx, "ExpertCount");
    PADDLE_ENFORCE_EQ(
        gate_idx_data_type, expert_count_data_type,
        platform::errors::InvalidArgument(
            "The dtype of the gate_idx and expert_count should be same"));
    PADDLE_ENFORCE_EQ(gate_idx_data_type, framework::proto::VarType::INT64,
                      platform::errors::InvalidArgument(
                          "The dtype of the gate_idx and expert_count should "
                          "be same as int64"));
    return framework::OpKernelType(gate_idx_data_type, ctx.device_context());
  }
};

class PruneGateByCapacityOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("GateIdx",
             "(Tensor), The gate_id sequence corresponding to the input data.");
    AddInput("ExpertCount",
             "(Tensor), The quantity value counted on the gate_id sequence of "
             "the input data.");
    AddAttr<int64_t>("n_expert", "The number of Experts on each worker")
        .SetDefault(0);
    AddAttr<int64_t>("n_worker", "The number of workers on the trainer")
        .SetDefault(0);

    AddOutput("NewGateIdx",
              "(Tensor), The gate_id sequence corresponding to the new input "
              "data after passing through prune.");
    // AddOutput(
    //     "ExpertCountOut",
    //     "(Tensor), The copy quantity value counted on the gate_id sequence of
    //     "
    //     "the input data.");

    AddComment(R"DOC(
prune_gate_by_capacity Operator.

This operator is used to prune gate by capacity(CUDA).

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(prune_gate_by_capacity, ops::PruneGateByCapacityOp,
                             ops::PruneGateByCapacityOpMaker);

REGISTER_OP_CPU_KERNEL(
    prune_gate_by_capacity,
    ops::PruneGateByCapacityCPUKernel<paddle::platform::CPUDeviceContext, int>,
    ops::PruneGateByCapacityCPUKernel<paddle::platform::CPUDeviceContext,
                                      int64_t>);
