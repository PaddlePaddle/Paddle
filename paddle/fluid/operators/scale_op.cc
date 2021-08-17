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

#include "paddle/fluid/operators/scale_op.h"
#include <string>
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace framework {
class InferShapeContext;
class OpDesc;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
namespace platform {
class CPUDeviceContext;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

class ScaleOp : public framework::OperatorWithKernel {
 public:
  ScaleOp(const std::string &type, const framework::VariableNameMap &inputs,
          const framework::VariableNameMap &outputs,
          const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "scale");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "scale");

    if (ctx->IsRuntime() && ctx->HasInput("ScaleTensor")) {
      auto scale = ctx->Inputs("ScaleTensor");
      PADDLE_ENFORCE_EQ(scale.size(), 1,
                        platform::errors::InvalidArgument(
                            "Input(ScaleTensor) size must be 1, "
                            "but received size is %d.",
                            scale.size()));
    }

    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ "Out");
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type =
        framework::OperatorWithKernel::IndicateVarDataType(ctx, "X");

#ifdef PADDLE_WITH_MKLDNN
    if (this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class ScaleOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input tensor of scale operator.");
    AddInput("ScaleTensor",
             "(Tensor) If provided, use this as "
             "scale factor, this has a higher priority than "
             "attr(scale), the shape of this tensor MUST BE 1.")
        .AsDispensable();
    AddOutput("Out", "(Tensor) Output tensor of scale operator.");
    AddComment(R"DOC(
**Scale operator**

Apply scaling and bias addition to the input tensor.

if bias_after_scale=True:

$$Out = scale*X + bias$$

else:

$$Out = scale*(X + bias)$$
)DOC");
    AddAttr<float>("scale", "The scaling factor of the scale operator.")
        .SetDefault(1.0);
    AddAttr<float>("bias", "The bias of the scale operator.").SetDefault(0.0);
    AddAttr<bool>(
        "bias_after_scale",
        "Apply bias addition after or before scaling. It is useful for "
        "numeric stability in some circumstances.")
        .SetDefault(true);
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
  }
};

class ScaleOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    ctx->SyncTypeAndDataType("X", "Out");
  }
};

template <typename T>
class ScaleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("scale");
    grad_op->SetInput("X", this->OutputGrad("Out"));
    if (this->HasInput("ScaleTensor") > 0) {
      grad_op->SetInput("ScaleTensor", this->Input("ScaleTensor"));
    }
    grad_op->SetOutput("Out", this->InputGrad("X"));
    grad_op->SetAttr("scale", this->GetAttr("scale"));
    grad_op->SetAttr("bias", 0.0f);
    grad_op->SetAttr("bias_after_scale", true);
    if (grad_op->HasAttr("use_mkldnn"))
      grad_op->SetAttr("use_mkldnn", this->GetAttr("use_mkldnn"));
  }
};

DECLARE_INPLACE_OP_INFERER(ScaleOpInplaceInferer, {"X", "Out"});
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(scale, ops::ScaleOp, ops::ScaleOpMaker,
                  ops::ScaleGradMaker<paddle::framework::OpDesc>,
                  ops::ScaleGradMaker<paddle::imperative::OpBase>,
                  ops::ScaleOpVarTypeInference, ops::ScaleOpInplaceInferer);
REGISTER_OP_CPU_KERNEL(
    scale, ops::ScaleKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ScaleKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ScaleKernel<paddle::platform::CPUDeviceContext,
                     paddle::platform::bfloat16>,
    ops::ScaleKernel<paddle::platform::CPUDeviceContext, uint8_t>,
    ops::ScaleKernel<paddle::platform::CPUDeviceContext, int8_t>,
    ops::ScaleKernel<paddle::platform::CPUDeviceContext, int16_t>,
    ops::ScaleKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ScaleKernel<paddle::platform::CPUDeviceContext, int64_t>);

REGISTER_OP_CUDA_KERNEL(
    scale,
    paddle::operators::ScaleKernel<paddle::platform::CUDADeviceContext, float>,
    paddle::operators::ScaleKernel<paddle::platform::CUDADeviceContext, double>,
    paddle::operators::ScaleKernel<paddle::platform::CUDADeviceContext,
                                   uint8_t>,
    paddle::operators::ScaleKernel<paddle::platform::CUDADeviceContext, int8_t>,
    paddle::operators::ScaleKernel<paddle::platform::CUDADeviceContext,
                                   int16_t>,
    paddle::operators::ScaleKernel<paddle::platform::CUDADeviceContext, int>,
    paddle::operators::ScaleKernel<paddle::platform::CUDADeviceContext,
                                   int64_t>,
    paddle::operators::ScaleKernel<paddle::platform::CUDADeviceContext,
                                   paddle::platform::float16>);
