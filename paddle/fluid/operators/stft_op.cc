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

#include "paddle/fluid/operators/stft_op.h"
#include "paddle/fluid/operators/spectral_helper.h"

namespace paddle {
namespace operators {
class StftOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "frame");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "frame");

    const int n_fft = ctx->Attrs().Get<int>("n_fft");
    const int hop_length = ctx->Attrs().Get<int>("hop_length");

    const auto x_dims = ctx->GetInputDim("X");
    const int x_rank = x_dims.size();
    const bool onesided = ctx->Attrs().Get<bool>("onesided");

    PADDLE_ENFORCE_EQ(
        x_rank, 2,
        platform::errors::InvalidArgument(
            "Input(X) of StftOp should be a tensor with shape [N, T], "
            "but got rank %s.",
            x_rank));
    PADDLE_ENFORCE_GT(
        hop_length, 0,
        platform::errors::InvalidArgument(
            "Attribute(hop_length) should be greater than 0, but got %s.",
            hop_length));

    int seq_length = x_dims[x_rank - 1];
    int n_frames = 1 + (seq_length - n_fft) / hop_length;

    PADDLE_ENFORCE_LE(n_fft, seq_length,
                      platform::errors::InvalidArgument(
                          "Attribute(frame_length) should be less equal than "
                          "sequence length, but got (%s) > (%s).",
                          n_fft, seq_length));

    std::vector<int64_t> output_shape;
    output_shape.push_back(x_dims[0]);
    if (onesided) {
      output_shape.push_back(n_fft / 2 + 1);
    } else {
      output_shape.push_back(n_fft);
    }
    output_shape.push_back(n_frames);

    ctx->SetOutputDim("Out", phi::make_ddim(output_shape));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    const auto in_dtype = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(in_dtype, ctx.GetPlace());
  }
};

class StftOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input waveforms with shape (N, T)");
    AddOutput("Out",
              "The complex STFT output tensor with shape (N, n_fft, "
              "num_frames) or (N, n_fft/2 + 1, num_frames)");
    AddAttr<int>("n_fft", "The number of input samples to perform FFT");
    AddAttr<int>("hop_length", "Number of samples between adjacent frames");
    AddAttr<bool>("normalized",
                  "Control whether to scale the output by 1/sqrt(n_fft)");
    AddAttr<bool>("onesided",
                  "Control whether to return half of the FFT output");
    AddComment(R"DOC(
      Short-time Fourier transform (STFT).
    )DOC");
  }
};

template <typename T>
class StftGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("stft_grad");
    grad_op->SetInput("X", this->Input("X"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

class StftGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    const auto out_grad_name = framework::GradVarName("Out");
    OP_INOUT_CHECK(ctx->HasInput(out_grad_name), "Input", out_grad_name,
                   "stft_grad");
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "stft_grad");

    const auto x_grad_name = framework::GradVarName("X");
    OP_INOUT_CHECK(ctx->HasOutput(x_grad_name), "Output", x_grad_name,
                   "stft_grad");

    ctx->ShareDim("X", /*->*/ x_grad_name);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    const auto in_dtype = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    const auto kernel_dtype = framework::ToRealType(in_dtype);
    return framework::OpKernelType(kernel_dtype, ctx.GetPlace());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(stft, ops::StftOp, ops::StftOpMaker,
                  ops::StftGradOpMaker<paddle::framework::OpDesc>,
                  ops::StftGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(stft_grad, ops::StftGradOp);

REGISTER_OP_CPU_KERNEL(
    stft, ops::StftKernel<paddle::platform::CPUDeviceContext, float>,
    ops::StftKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(
    stft_grad, ops::StftGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::StftGradKernel<paddle::platform::CPUDeviceContext, double>);
