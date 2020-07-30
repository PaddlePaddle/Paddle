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

#include "paddle/fluid/operators/cross_norm_hadamard_op.h"
#include <string>

namespace paddle {
namespace operators {

class CrossNormHadamardOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "X", "CrossNormHadamard");
    OP_INOUT_CHECK(ctx->HasInput("SummaryInput"), "Input", "SummaryInput",
                   "CrossNormHadamard");

    OP_INOUT_CHECK(ctx->HasOutput("CudaMeans"), "Output", "CudaMeans",
                   "CrossNormHadamard");
    OP_INOUT_CHECK(ctx->HasOutput("CudaScales"), "Output", "CudaScales",
                   "CrossNormHadamard");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "CrossNormHadamard");

    auto fields_num = ctx->Attrs().Get<int64_t>("fields_num");
    auto embed_dim = ctx->Attrs().Get<int64_t>("embed_dim");

    auto cols = (embed_dim * 3 + 1) * fields_num;
    auto input_dims = ctx->GetInputDim("Input");
    auto summary_dims = ctx->GetInputDim("SummaryInput");

    PADDLE_ENFORCE_EQ(
        cols, summary_dims[1],
        platform::errors::InvalidArgument("Input(SummaryInput) should be [%d],"
                                          "but now it is [%d]",
                                          cols, summary_dims[0]));

    PADDLE_ENFORCE_EQ(embed_dim * 2 * fields_num, input_dims[1],
                      platform::errors::InvalidArgument(
                          "Input(Input) should be [%d],"
                          "but now it is [%d]",
                          embed_dim * 2 * fields_num, input_dims[1]));
    ctx->SetOutputDim("Out", {input_dims[0], cols});
    ctx->SetOutputDim("CudaMeans", {1, cols});
    ctx->SetOutputDim("CudaScales", {1, cols});

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

class CrossNormHadamardGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Input"), true,
        platform::errors::InvalidArgument("Input should not be null"));
    PADDLE_ENFORCE_EQ(ctx->HasInput("SummaryInput"), true,
                      platform::errors::InvalidArgument(
                          "Input(SummaryInput) should not be null"));

    ctx->SetOutputDim(framework::GradVarName("Input"),
                      ctx->GetInputDim("Input"));
    ctx->SetOutputDim(framework::GradVarName("SummaryInput"),
                      ctx->GetInputDim("SummaryInput"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

class CrossNormHadamardOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input",
             "(Tensor) Input tensor of cross_norm_hadamard_op operator.");
    AddInput("SummaryInput",
             "(Tensor) Input tensor of cross_norm_hadamard_op operator.");
    AddAttr<int64_t>("fields_num", "(int64_t) the fields_num").SetDefault(2);
    AddAttr<int64_t>("embed_dim", "(int64_t) the embed_dim").SetDefault(1);
    AddAttr<float>(
        "summary_decay_rate",
        "(float, default 0.9999999) The decay rate when update the summary")
        .SetDefault(0.9999999);
    AddAttr<float>("epsilon", "")
        .SetDefault(1e-4)
        .AddCustomChecker([](const float& epsilon) {
          PADDLE_ENFORCE(epsilon >= 0.0f && epsilon <= 0.001f,
                         "'epsilon' should be between 0.0 and 0.001.");
        });
    AddOutput("Out", "Output tensor of cross_norm_hadamard_op operator.");
    AddOutput("CudaMeans", "Output tensor of cross_norm_hadamard_op operator.");
    AddOutput("CudaScales",
              "Output tensor of cross_norm_hadamard_op operator.");

    AddComment(R"DOC(
CrossNormHadamard Operator.
Notice: It currently supports GPU device.
This Op exists in contrib, which means that it is not shown to the public.
)DOC");
  }
};

template <typename T>
class CrossNormHadamardGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("cross_norm_hadamard_grad");

    op->SetInput("Input", this->Input("Input"));
    op->SetInput("SummaryInput", this->Input("SummaryInput"));
    op->SetInput("Out", this->Output("Out"));
    op->SetInput("CudaMeans", this->Output("CudaMeans"));
    op->SetInput("CudaScales", this->Output("CudaScales"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));

    op->SetOutput("SummaryInput", this->Input("SummaryInput"));
    op->SetOutput(framework::GradVarName("Input"), this->InputGrad("Input"));
    op->SetOutput(framework::GradVarName("SummaryInput"),
                  this->InputGrad("SummaryInput"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    cross_norm_hadamard, ops::CrossNormHadamardOp,
    ops::CrossNormHadamardOpMaker,
    ops::CrossNormHadamardGradOpMaker<paddle::framework::OpDesc>,
    ops::CrossNormHadamardGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(cross_norm_hadamard_grad, ops::CrossNormHadamardGradOp);

REGISTER_OP_CPU_KERNEL(
    cross_norm_hadamard,
    ops::CrossNormHadamardKernel<paddle::platform::CPUDeviceContext, float>,
    ops::CrossNormHadamardKernel<paddle::platform::CPUDeviceContext, double>);
