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

#include "paddle/fluid/operators/random_routing_op.h"

namespace paddle {
namespace operators {

class RandomRoutingOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Prob"), "Input", "Porb", "RandomRouting");
    OP_INOUT_CHECK(ctx->HasInput("TopK_Value"), "Input", "TopKValue",
                   "RandomRouting");
    OP_INOUT_CHECK(ctx->HasInput("TopK_Idx"), "Input", "TopKIdx",
                   "RandomRouting");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "RandomRouting");

    // check dims

    auto topk_val_dims = ctx->GetInputDim("TopK_Value");
    auto prob_dims = ctx->GetInputDim("Prob");
    auto topk_idx_dims = ctx->GetInputDim("TopK_Idx");

    PADDLE_ENFORCE_EQ(prob_dims[0], topk_val_dims[0],
                      platform::errors::InvalidArgument(
                          "Output(Out) of ScatterNdAddOp should not be null."));

    PADDLE_ENFORCE_EQ(topk_idx_dims[1], topk_val_dims[1],
                      platform::errors::InvalidArgument(
                          "Output(Out) of ScatterNdAddOp should not be null."));

    PADDLE_ENFORCE_EQ(topk_idx_dims[0], topk_val_dims[0],
                      platform::errors::InvalidArgument(
                          "Output(Out) of ScatterNdAddOp should not be null."));

    ctx->SetOutputDim("Out", topk_idx_dims);
    ctx->ShareLoD("TopK_Idx", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    // the dtype of the gate_idx should be same as int64
    const auto topk_idx_dtype =
        OperatorWithKernel::IndicateVarDataType(ctx, "TopK_Idx");
    PADDLE_ENFORCE_EQ(topk_idx_dtype, framework::proto::VarType::INT64,
                      platform::errors::InvalidArgument(
                          "The dtype of the topk_idx_dtype should be int64"));

    const auto& topk_value_type =
        OperatorWithKernel::IndicateVarDataType(ctx, "TopK_Value");
    return framework::OpKernelType(topk_value_type, ctx.GetPlace());
  }
};

class RandomRoutingOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Prob", "(Tensor) The input Prob index tensor.");
    AddInput("TopK_Value", "(Tensor) The input TopK_Value index tensor.");
    AddInput("TopK_Idx", "(Tensor) The input TopK_Idx index tensor.");
    AddOutput("Out", "(Tensor) The output random routing tensor.");
    AddComment(R"DOC(expert_count Operator random routing.)DOC");
  }
};

DECLARE_INPLACE_OP_INFERER(RandomRoutingInplaceInferer, {"TopK_Idx", "Out"});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(
    random_routing, ops::RandomRoutingOp, ops::RandomRoutingOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    ops::RandomRoutingInplaceInferer)

REGISTER_OP_CPU_KERNEL(random_routing, ops::RandomRoutingOpCPUKernel<float>,
                       ops::RandomRoutingOpCPUKernel<double>);
