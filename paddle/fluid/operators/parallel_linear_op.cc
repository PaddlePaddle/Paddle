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

#include "paddle/fluid/operators/parallel_linear_op.h"

namespace paddle {
namespace operators {

class ParallelLinearOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    // global_input_buf
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ParallelLinear");
    // Weight
    OP_INOUT_CHECK(ctx->HasInput("W"), "Input", "W", "ParallelLinear");
    // Bias
    OP_INOUT_CHECK(ctx->HasInput("Bias"), "Input", "Bias", "ParallelLinear");

    // fwd_expert_count
    // OP_INOUT_CHECK(ctx->HasInput("Expert_Count"), "Input", "Expert_Count",
    //  "ParallelLinear");

    // global_output_buf
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "ParallelLinear");

    auto x_dims = ctx->GetInputDim("X");
    auto w_dims = ctx->GetInputDim("W");
    auto b_dims = ctx->GetInputDim("Bias");

    // auto expert_count_dims = ctx->GetInputDim("Expert_Count");
    auto expert_count = ctx->Attrs().Get<std::vector<int>>("expert_count");
    auto expert_count_dims = expert_count.size();

    PADDLE_ENFORCE_EQ(x_dims.size(), 2,
                      platform::errors::InvalidArgument(
                          "X's shape size should be 2, "
                          "but received the size of Input(x)'s shape is %d",
                          x_dims.size()));

    PADDLE_ENFORCE_EQ(w_dims.size(), 3,
                      platform::errors::InvalidArgument(
                          "X's shape size should be 3, "
                          "but received the size of Input(w)'s shape is %d.",
                          x_dims.size()));

    PADDLE_ENFORCE_EQ(b_dims.size(), 2,
                      platform::errors::InvalidArgument(
                          "X's shape size should be 2, "
                          "but received the size of Input(bias)'s shape is %d.",
                          x_dims.size()));

    PADDLE_ENFORCE_EQ(x_dims[1], w_dims[1],
                      platform::errors::InvalidArgument(
                          "X's shape[1] should be equal to W's shape[1], "
                          "but received X's shape[1] = %d, W's shape[1] = %d.",
                          x_dims[1], w_dims[1]));

    PADDLE_ENFORCE_EQ(
        expert_count_dims, w_dims[0],
        platform::errors::InvalidArgument(
            "Expert_Count's shape[0] should be equal to W's shape[0], "
            "but received Expert_Count's shape[0] = %d, W's shape[0] = %d.",
            expert_count_dims, w_dims[0]));

    PADDLE_ENFORCE_EQ(
        w_dims[2], b_dims[1],
        platform::errors::InvalidArgument(
            "W's shape[2] should be equal to Bias's shape[1], "
            "but received W's shape[1] = %d, Bias's shape[1] = %d.",
            w_dims[2], b_dims[1]));

    ctx->SetOutputDim("Out", {x_dims[0], w_dims[2]});
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class ParallelLinearOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input tensor of batch_fc_op operator.");
    AddInput("W", "(Tensor) Input tensor of batch_fc_op operator.");
    AddInput("Bias", "(Tensor) Input tensor of batch_fc_op operator.");
    // AddInput("Expert_Count", "(Tensor) Input tensor of batch_fc_op
    // operator.");

    AddOutput("Out", "Output tensor of batch_fc_op operator.");

    AddAttr<std::vector<int>>("expert_count", "The expert count.")
        .SetDefault(std::vector<int>());

    AddComment(R"DOC(
ParallelLinearOp Operator.
Notice: It currently supports GPU device.
This Op exists in contrib, which means that it is not shown to the public.
)DOC");
  }
};

class ParallelLinearGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::InvalidArgument("Input should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("W"), true,
        platform::errors::InvalidArgument("Input(W) should not be null"));

    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    ctx->SetOutputDim(framework::GradVarName("W"), ctx->GetInputDim("W"));
    ctx->SetOutputDim(framework::GradVarName("Bias"), ctx->GetInputDim("Bias"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

template <typename T>
class ParallelLinearGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("parallel_linear_grad");

    op->SetInput("X", this->Input("X"));
    op->SetInput("W", this->Input("W"));
    op->SetInput("Bias", this->Input("Bias"));
    // op->SetInput("Expert_Count", this->Input("Expert_Count"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("W"), this->InputGrad("W"));
    op->SetOutput(framework::GradVarName("Bias"), this->InputGrad("Bias"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(ParallelLinearGradOpNoNeedBufferVarsInferer,
                                    "Bias");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(parallel_linear, ops::ParallelLinearOp,
                  ops::ParallelLinearOpMaker,
                  ops::ParallelLinearGradOpMaker<paddle::framework::OpDesc>,
                  ops::ParallelLinearGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(parallel_linear_grad, ops::ParallelLinearGradOp,
                  ops::ParallelLinearGradOpNoNeedBufferVarsInferer);

REGISTER_OP_CPU_KERNEL(
    parallel_linear,
    ops::ParallelLinearOpCPUKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ParallelLinearOpCPUKernel<paddle::platform::CPUDeviceContext, double>);
