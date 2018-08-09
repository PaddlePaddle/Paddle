/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/dropout_op.h"

namespace paddle {
namespace operators {

class DropoutOp : public framework::OperatorWithKernel {
  using Tensor = framework::LoDTensor;

 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) must not be null.");

    auto x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", x_dims);
    if (ctx->Attrs().Get<bool>("is_test") == false) {
      if (ctx->Attrs().Get<bool>("fix_seed")) {
        PADDLE_ENFORCE(ctx->HasInput("SeedIn"),
                       "Input(SeedIn) must not be null when fix_seed is true.");
        PADDLE_ENFORCE(
            ctx->HasOutput("SeedOut"),
            "Output(SeedOut) must not be null when fix_seed is true.");
      }
      ctx->SetOutputDim("Mask", x_dims);
    }
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("X")->type()), ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const framework::Tensor& tensor,  // NOLINT
      const framework::OpKernelType& expected_kernel_type) const override {
    // Prevent data transform of Input(SeedIn) from CPUPlace to CUDAPlace
    return var_name == "SeedIn"
               ? expected_kernel_type
               : framework::OperatorWithKernel::GetKernelTypeForVar(
                     var_name, tensor, expected_kernel_type);
  }
};

class DropoutOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input of dropout op.");
    AddInput("SeedIn",
             "The input random seed of dropout op when fix_seed is true. It "
             "may be changed after each iterations")
        .AsDispensable();
    AddOutput("Out", "The output of dropout op.");
    AddOutput("Mask", "The random sampled dropout mask.").AsIntermediate();
    AddOutput("SeedOut",
              "The output random seed of dropout op when fix_seed is true. It "
              "should share memory with Input(SeedIn)")
        .AsDispensable();

    AddAttr<float>("dropout_prob", "Probability of setting units to zero.")
        .SetDefault(.5f)
        .AddCustomChecker([](const float& drop_p) {
          PADDLE_ENFORCE(drop_p >= 0.0f && drop_p <= 1.0f,
                         "'dropout_prob' must be between 0.0 and 1.0.");
        });
    AddAttr<bool>("is_test", "True if in test phase.").SetDefault(false);
    AddAttr<bool>("fix_seed",
                  "A flag indicating whether to use a fixed seed to generate "
                  "random mask. NOTE: DO NOT set this flag to true in "
                  "training. Setting this flag to true is only useful in "
                  "unittest or for debug that always the same output units "
                  "will be dropped.")
        .SetDefault(false);
    AddAttr<int>("startup_seed",
                 "The startup seed when fix_seed is true and Input(SeedIn) has "
                 "not been initialized.")
        .SetDefault(0);
    AddComment(R"DOC(
Dropout Operator.

Dropout refers to randomly dropping out units in a nerual network. It is a
regularization technique for reducing overfitting by preventing neuron
co-adaption during training. The dropout operator randomly set (according to
the given dropout probability) the outputs of some units to zero, while others
are set equal to their corresponding inputs.

)DOC");
  }
};

class DropoutOpGrad : public framework::OperatorWithKernel {
  using Tensor = framework::LoDTensor;

 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->Attrs().Get<bool>("is_test"), false,
                      "GradOp is only callable when is_test is false");

    PADDLE_ENFORCE(ctx->HasInput("Mask"), "Mask must not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) must not be null.");

    auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    auto mask_dims = ctx->GetInputDim("Mask");
    PADDLE_ENFORCE_EQ(
        out_dims, mask_dims,
        "Dimensions of Input(Mask) and Input(Out@GRAD) must be the same.");

    ctx->SetOutputDim(framework::GradVarName("X"), mask_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(
            ctx.Input<Tensor>(framework::GradVarName("Out"))->type()),
        ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const framework::Tensor& tensor,  // NOLINT
      const framework::OpKernelType& expected_kernel_type) const override {
    // Prevent data transform of Input(SeedIn) from CPUPlace to CUDAPlace
    return (var_name == "SeedIn" || var_name == "SeedOut")
               ? expected_kernel_type
               : framework::OperatorWithKernel::GetKernelTypeForVar(
                     var_name, tensor, expected_kernel_type);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(dropout, ops::DropoutOp, ops::DropoutOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(dropout_grad, ops::DropoutOpGrad);
REGISTER_OP_CPU_KERNEL(dropout, ops::CPUDropoutKernel<float>);
REGISTER_OP_CPU_KERNEL(
    dropout_grad,
    ops::DropoutGradKernel<paddle::platform::CPUDeviceContext, float>);
