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

#include "paddle/fluid/operators/sequence_softmax_op.h"
#include <string>

namespace paddle {
namespace operators {

class SequenceSoftmaxOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of SequenceSoftmaxOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of SequenceSoftmaxOp should not be null.");
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    // choose cudnn kernel if the runtime supported.
    bool use_cudnn = ctx.Attr<bool>("use_cudnn");
    bool runtime_cudnn_support = false;
#ifdef PADDLE_WITH_CUDA
    if (platform::is_gpu_place(ctx.GetPlace())) {
      auto& dev_ctx =
          ctx.template device_context<platform::CUDADeviceContext>();
      runtime_cudnn_support = dev_ctx.cudnn_handle() != nullptr ? true : false;
    }
#endif
    framework::LibraryType library_ = framework::LibraryType::kPlain;
    if (use_cudnn && runtime_cudnn_support) {
      library_ = framework::LibraryType::kCUDNN;
    }
    std::string data_format = ctx.Attr<std::string>("data_format");
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("X")->type()), ctx.GetPlace(),
        framework::StringToDataLayout(data_format), library_);
  }
};

class SequenceSoftmaxOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(LoDTensor) 1-D or 2-D input LoDTensor with the 2-nd dimension "
             "of length 1.");
    AddOutput("Out",
              "(LoDTensor) 1-D or 2-D output LoDTensor with the 2-nd dimension "
              "of length 1.");
    AddAttr<bool>(
        "use_cudnn",
        "(bool, default false) Only used in cudnn kernel, need install cudnn")
        .SetDefault(false);
    AddAttr<std::string>(
        "data_format",
        "(string, default NCHW) Only used in "
        "An optional string from: \"NHWC\", \"NCHW\". "
        "Defaults to \"NHWC\". Specify the data format of the output data, "
        "the input will be transformed automatically. ")
        .SetDefault("AnyLayout");
    AddComment(R"DOC(
Sequence Softmax Operator.

SequenceSoftmaxOp computes the softmax activation among all time-steps for each
sequence. The dimension of each time-step should be 1. Thus, the shape of
input Tensor can be either [N, 1] or [N], where N is the sum of the length
of all sequences.

The algorithm works as follows:

    for i-th sequence in a mini-batch:

$$
Out(X[lod[i]:lod[i+1]], :) = \
\frac{\exp(X[lod[i]:lod[i+1], :])} \
{\sum(\exp(X[lod[i]:lod[i+1], :]))}
$$

For example, for a mini-batch of 3 sequences with variable-length,
each containing 2, 3, 2 time-steps, the lod of which is [0, 2, 5, 7],
then softmax will be computed among X[0:2, :], X[2:5, :], X[5:7, :]
and N turns out to be 7.

)DOC");
  }
};

class SequenceSoftmaxGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Out"),
                   "Input(Out) of SequenceSoftmaxGradOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput(framework::GradVarName("Out")),
        "Input(Out@GRAD) of SequenceSoftmaxGradOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of SequenceSoftmaxOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                   "Output(X@GRAD) of SequenceSoftmaxOp should not be null.");

    PADDLE_ENFORCE_EQ(
        ctx->GetInputDim("Out"),
        ctx->GetInputDim(framework::GradVarName("Out")),
        "Input(Out) and Input(Out@GRAD) of SequenceSoftmaxGradOp should be of "
        "the same shape.");

    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    // choose cudnn kernel if the runtime supported.
    bool use_cudnn = ctx.Attr<bool>("use_cudnn");
    bool runtime_cudnn_support = false;
#ifdef PADDLE_WITH_CUDA
    if (platform::is_gpu_place(ctx.GetPlace())) {
      auto& dev_ctx =
          ctx.template device_context<platform::CUDADeviceContext>();
      runtime_cudnn_support = dev_ctx.cudnn_handle() != nullptr ? true : false;
    }
#endif
    framework::LibraryType library_ = framework::LibraryType::kPlain;
    if (use_cudnn && runtime_cudnn_support) {
      library_ = framework::LibraryType::kCUDNN;
    }
    std::string data_format = ctx.Attr<std::string>("data_format");
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("X")->type()), ctx.GetPlace(),
        framework::StringToDataLayout(data_format), library_);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(sequence_softmax, ops::SequenceSoftmaxOp,
                  ops::SequenceSoftmaxOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(sequence_softmax_grad, ops::SequenceSoftmaxGradOp);
REGISTER_OP_CPU_KERNEL(
    sequence_softmax,
    ops::SequenceSoftmaxKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SequenceSoftmaxKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    sequence_softmax_grad,
    ops::SequenceSoftmaxGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SequenceSoftmaxGradKernel<paddle::platform::CPUDeviceContext, double>);
