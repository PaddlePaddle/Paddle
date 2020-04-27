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

#include "paddle/fluid/operators/batch_fc_op.h"
#include <string>

namespace paddle {
namespace operators {

class BatchFCOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Input"), true,
        platform::errors::InvalidArgument(
            "X(Input) of Batch Fully Connected should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Out"), true,
        platform::errors::InvalidArgument(
            "Out(Output) of Batch Fully Connected should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("W"), true,
        platform::errors::InvalidArgument(
            "W(Input) of Batch Fully Connected should not be null."));

    auto input_dims = ctx->GetInputDim("Input");
    auto w_dims = ctx->GetInputDim("W");

    PADDLE_ENFORCE_EQ(input_dims.size(), 3,
                      platform::errors::InvalidArgument(
                          "Input of BatchFCOp should have 3D."));
    PADDLE_ENFORCE_EQ(w_dims.size(), 3, platform::errors::InvalidArgument(
                                            "W of BatchFCOp should have 3D."));
    PADDLE_ENFORCE_EQ(
        input_dims[0], w_dims[0],
        platform::errors::InvalidArgument(
            "Input.dim[0] and W.dim[0] of BatchFCOp should be same."));
    PADDLE_ENFORCE_EQ(
        input_dims[2], w_dims[1],
        platform::errors::InvalidArgument(
            "Input.dim[2] and W.dim[1] of BatchFCOp should be same."));

    auto bias_dims = ctx->GetInputDim("Bias");
    PADDLE_ENFORCE_EQ(bias_dims[0], input_dims[0],
                      platform::errors::InvalidArgument(
                          "Bias.dim[0] should be same as input.dim[0]."));
    PADDLE_ENFORCE_EQ(bias_dims[1], w_dims[2],
                      platform::errors::InvalidArgument(
                          "Bias.dim[1] should be same as input.dim[2]."));

    ctx->SetOutputDim("Out", {input_dims[0], input_dims[1], w_dims[2]});
    ctx->ShareLoD("Input", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
        ctx.device_context());
  }
};

class BatchFCGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Input"), true,
        platform::errors::InvalidArgument("Input should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("W"), true,
        platform::errors::InvalidArgument("Input(W) should not be null"));

    ctx->SetOutputDim(framework::GradVarName("Input"),
                      ctx->GetInputDim("Input"));
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

class BatchFCOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "(Tensor) Input tensor of batch_fc_op operator.");
    AddInput("W", "(Tensor) Input tensor of batch_fc_op operator.");
    AddInput("Bias", "(Tensor) Input tensor of batch_fc_op operator.");
    AddOutput("Out", "Output tensor of batch_fc_op operator.");
    AddComment(R"DOC(
BatchFC Operator.
Notice: It currently supports GPU device.
This Op exists in contrib, which means that it is not shown to the public.
)DOC");
  }
};

template <typename T>
class BatchFCGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("batch_fc_grad");

    op->SetInput("Input", this->Input("Input"));
    op->SetInput("W", this->Input("W"));
    op->SetInput("Bias", this->Input("Bias"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));

    op->SetOutput(framework::GradVarName("Input"), this->InputGrad("Input"));
    op->SetOutput(framework::GradVarName("W"), this->InputGrad("W"));
    op->SetOutput(framework::GradVarName("Bias"), this->InputGrad("Bias"));
    op->SetAttrMap(this->Attrs());
  }
};
DECLARE_NO_NEED_BUFFER_VARS_INFERER(BatchFCGradOpNoNeedBufferVarsInference,
                                    "Bias");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(batch_fc, ops::BatchFCOp, ops::BatchFCOpMaker,
                  ops::BatchFCGradOpMaker<paddle::framework::OpDesc>,
                  ops::BatchFCGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(batch_fc_grad, ops::BatchFCGradOp,
                  ops::BatchFCGradOpNoNeedBufferVarsInference);

REGISTER_OP_CPU_KERNEL(
    batch_fc, ops::BatchFCKernel<paddle::platform::CPUDeviceContext, float>,
    ops::BatchFCKernel<paddle::platform::CPUDeviceContext, double>);
