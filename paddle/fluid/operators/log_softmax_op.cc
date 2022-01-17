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

#include "paddle/fluid/operators/log_softmax_op.h"
#include <string>
#include <unordered_map>
#include "paddle/fluid/operators/common_infer_shape_functions.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"

namespace paddle {
namespace operators {

class LogSoftmaxOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    return UnaryOpUnchangedInferShapeCheckAxis(ctx);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    // choose cudnn kernel if the runtime supported.
    framework::LibraryType library{framework::LibraryType::kPlain};
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (platform::CanCUDNNBeUsed(ctx)) {
      library = framework::LibraryType::kCUDNN;
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.device_context(),
                                   layout, library);
  }
};

class LogSoftmaxOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input tensor of softmax, "
             "whose dimension :attr:`axis` is the input_feature_dimensions.");
    AddOutput("Out", "The normalized values with the same shape as X.");
    AddAttr<int>("axis",
                 "The dimension index of Input(x) to perform log_softmax,"
                 "default -1 for last dimension")
        .SetDefault(-1);
    AddAttr<bool>(
        "use_cudnn",
        "(bool, default false) Only used in cudnn kernel, need install cudnn")
        .SetDefault(false)
        .AsExtra();
    AddComment(R"DOC(
LogSoftmax Operator.

)DOC");
  }
};

class LogSoftmaxOpInferVarType
    : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Out"}};
    return m;
  }
};

class LogSoftmaxGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Out"), "Input", "Out", "log_softmax_grad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@grad", "log_softmax_grad");
    PADDLE_ENFORCE_EQ(
        ctx->GetInputDim("Out"),
        ctx->GetInputDim(framework::GradVarName("Out")),
        platform::errors::InvalidArgument("Input(Out) and its gradients "
                                          "should have the same shape."));

    ctx->SetOutputDim(framework::GradVarName("X"),
                      ctx->GetInputDim(framework::GradVarName("Out")));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    // choose cudnn kernel if the runtime supported.
    framework::LibraryType library{framework::LibraryType::kPlain};
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (platform::CanCUDNNBeUsed(ctx)) {
      library = framework::LibraryType::kCUDNN;
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.device_context(),
                                   layout, library);
  }
};

template <typename T>
class LogSoftmaxGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("log_softmax_grad");
    op->SetInput("Out", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(log_softmax, ops::LogSoftmaxOp, ops::LogSoftmaxOpMaker,
                  ops::LogSoftmaxOpInferVarType,
                  ops::LogSoftmaxGradOpMaker<paddle::framework::OpDesc>,
                  ops::LogSoftmaxGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(log_softmax_grad, ops::LogSoftmaxGradOp);

REGISTER_OP_CPU_KERNEL(
    log_softmax,
    ops::LogSoftmaxKernel<paddle::platform::CPUDeviceContext, float>,
    ops::LogSoftmaxKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    log_softmax_grad,
    ops::LogSoftmaxGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::LogSoftmaxGradKernel<paddle::platform::CPUDeviceContext, double>);
