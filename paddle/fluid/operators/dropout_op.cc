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
#include <memory>
#include <string>

namespace paddle {
namespace operators {

using framework::Tensor;

class DropoutOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Dropout");

    auto x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", x_dims);
    if (ctx->Attrs().Get<bool>("is_test") == false) {
      ctx->SetOutputDim("Mask", x_dims);
    }
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    if (var_name == "Seed") {
      VLOG(10) << "var_name:" << var_name
               << " does not need to transform in dropout op";
      return expected_kernel_type;
    }

    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

class DropoutOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input of dropout op.");
    AddInput("Seed",
             "The seed of dropout op, it has higher priority than the attr "
             "fix_seed and seed")
        .AsDispensable()
        .AsExtra();
    AddOutput("Out", "The output of dropout op.");
    AddOutput("Mask", "The random sampled dropout mask.")
        .AsIntermediate()
        .AsExtra();

    AddAttr<float>("dropout_prob", "Probability of setting units to zero.")
        .SetDefault(.5f)
        .AddCustomChecker([](const float& drop_p) {
          PADDLE_ENFORCE_EQ(drop_p >= 0.0f && drop_p <= 1.0f, true,
                            platform::errors::InvalidArgument(
                                "'dropout_prob' must be between 0.0 and 1.0."));
        });
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false);
    AddAttr<bool>("fix_seed",
                  "A flag indicating whether to use a fixed seed to generate "
                  "random mask. NOTE: DO NOT set this flag to true in "
                  "training. Setting this flag to true is only useful in "
                  "unittest or for debug that always the same output units "
                  "will be dropped.")
        .SetDefault(false)
        .AsExtra();
    AddAttr<int>("seed", "Dropout random seed.").SetDefault(0).AsExtra();
    AddAttr<std::string>(
        "dropout_implementation",
        "[\"downgrade_in_infer\"|\"upscale_in_train\"]"
        "There are two kinds of ways to implement dropout"
        "(the mask below is a tensor have the same shape with input"
        "the value of mask is 0 or 1, the ratio of 0 is dropout_prob)"
        "1. downgrade_in_infer(default), downgrade the outcome at inference "
        "time"
        "   train: out = input * mask"
        "   inference: out = input * (1.0 - dropout_prob)"
        "2. upscale_in_train, upscale the outcome at training time, do nothing "
        "in inference"
        "   train: out = input * mask / ( 1.0 - dropout_prob )"
        "   inference: out = input"
        "   dropout op can be removed from the program. the program will be "
        "efficient")
        .SetDefault("downgrade_in_infer")
        .AddCustomChecker([](const std::string& type) {
          PADDLE_ENFORCE_EQ(
              type == "downgrade_in_infer" || type == "upscale_in_train", true,
              platform::errors::InvalidArgument(
                  "dropout_implementation can only be downgrade_in_infer or "
                  "upscale_in_train"));
        });

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
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Mask"), "Input", "Mask", "DropoutGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "DropoutGrad");

    auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));

    ctx->SetOutputDim(framework::GradVarName("X"), out_dims);
    ctx->ShareLoD(framework::GradVarName("Out"),
                  /*->*/ framework::GradVarName("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace());
  }
};

template <typename T>
class DropoutGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("dropout_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput("Mask", this->Output("Mask"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(dropout, ops::DropoutOp, ops::DropoutOpMaker,
                  ops::DropoutGradOpMaker<paddle::framework::OpDesc>,
                  ops::DropoutGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(dropout_grad, ops::DropoutOpGrad);
REGISTER_OP_CPU_KERNEL(
    dropout, ops::CPUDropoutKernel<paddle::platform::CPUDeviceContext, float>,
    ops::CPUDropoutKernel<paddle::platform::CPUDeviceContext, double>,
    ops::CPUDropoutKernel<paddle::platform::CPUDeviceContext,
                          paddle::platform::bfloat16>);
REGISTER_OP_CPU_KERNEL(
    dropout_grad,
    ops::DropoutGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::DropoutGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::DropoutGradKernel<paddle::platform::CPUDeviceContext,
                           paddle::platform::bfloat16>);
