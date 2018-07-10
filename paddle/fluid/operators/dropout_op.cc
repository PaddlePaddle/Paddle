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
#include "paddle/fluid/framework/var_type.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class DropoutOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) must not be null.");

    auto x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", x_dims);

    bool use_cudnn = ctx->Attrs().Get<bool>("use_cudnn");
    bool is_test = ctx->Attrs().Get<bool>("is_test");

    if (!is_test) {
      if (use_cudnn) {
        // Since we cannot access DeviceContext here,
        // dims of States cannot be inferred by calling
        // cudnnDropoutGetStatesSize
        PADDLE_ENFORCE(ctx->HasOutput("States"),
                       "Output(States) must not be null when use cudnnDropout");

        // Since we cannot get data_type here,
        // dims of ReserveSpace cannot be inferred by calling
        // cudnnDropoutGetReserveSpaceSize
        PADDLE_ENFORCE(
            ctx->HasOutput("ReserveSpace"),
            "Output(ReserveSpace) must not be null when use cudnnDropout");
      } else {
        ctx->SetOutputDim("Mask", x_dims);
      }
    }

    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto place = ctx.GetPlace();
    bool use_cudnn = ctx.Attr<bool>("use_cudnn");
    if (use_cudnn) {
      PADDLE_ENFORCE(platform::is_gpu_place(place),
                     "cudnn is not supported in CPUPlace");
    }
    framework::LibraryType library =
        (use_cudnn ? framework::LibraryType::kCUDNN
                   : framework::LibraryType::kPlain);
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    auto input_data_type =
        framework::ToDataType(ctx.Input<Tensor>("X")->type());
    return framework::OpKernelType(input_data_type, place, layout, library);
  }
};

class DropoutOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input of dropout op.");
    AddOutput("Out", "The output of dropout op.");
    AddOutput("Mask",
              "The random sampled dropout mask when use_cudnn is false.")
        .AsDispensable()
        .AsIntermediate();
    AddOutput("States", "The random generator states when use_cudnn is true.")
        .AsDispensable()
        .AsIntermediate();
    AddOutput("ReserveSpace", "The reserve space when use_cudnn is true.")
        .AsDispensable()
        .AsIntermediate();

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
    AddAttr<int>("seed", "Dropout random seed.").SetDefault(0);
    AddAttr<bool>("use_cudnn",
                  "Whether to use cudnn kernel."
                  "Notice that when use_cudnn, the approximately dropout_prob "
                  "fraction of X values will be replaces by 0, and the rest "
                  "will be scaled by 1/(1-dropout_prob).")
        .SetDefault(false);

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
    PADDLE_ENFORCE_EQ(ctx->Attrs().Get<bool>("is_test"), false,
                      "GradOp is only callable when is_test is false");

    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) must not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Mask"), "Mask must not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) must not be null.");

    auto x_dims = ctx->GetInputDim("X");
    auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    PADDLE_ENFORCE_EQ(x_dims, out_dims,
                      "Dimensions of Input(X) and Out@Grad must be the same.");
    auto mask_dims = ctx->GetInputDim("Mask");
    PADDLE_ENFORCE_EQ(x_dims, mask_dims,
                      "Dimensions of Input(X) and Mask must be the same.");

    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto place = ctx.GetPlace();
    bool use_cudnn = ctx.Attr<bool>("use_cudnn");
    if (use_cudnn) {
      PADDLE_ENFORCE(platform::is_gpu_place(place),
                     "cudnn is not supported in CPUPlace");
    }
    framework::LibraryType library =
        (use_cudnn ? framework::LibraryType::kCUDNN
                   : framework::LibraryType::kPlain);
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;

    // FIXME(zengjinle): when use_cudnn is false,
    // dropout GPU kernel supports FP16 and FP32,
    // but dropout_grad GPU kernel only supports FP32
    auto input_data_type =
        framework::ToDataType(ctx.Input<Tensor>("X")->type());
    if (platform::is_gpu_place(place) && !use_cudnn) {
      input_data_type = framework::proto::VarType::FP32;
    }
    return framework::OpKernelType(input_data_type, place, layout, library);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(dropout, ops::DropoutOp, ops::DropoutOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(dropout_grad, ops::DropoutOpGrad);
REGISTER_OP_CPU_KERNEL(
    dropout, ops::CPUDropoutKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    dropout_grad,
    ops::DropoutGradKernel<paddle::platform::CPUDeviceContext, float>);
