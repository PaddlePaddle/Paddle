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

#include "paddle/fluid/operators/frame_op.h"

namespace paddle {
namespace operators {

class FrameOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "frame");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "frame");

    const int frame_length = ctx->Attrs().Get<int>("frame_length");
    const int hop_length = ctx->Attrs().Get<int>("hop_length");
    const int axis = ctx->Attrs().Get<int>("axis");

    const auto x_dims = ctx->GetInputDim("X");
    const int x_rank = x_dims.size();

    PADDLE_ENFORCE_GE(
        x_rank, 1, platform::errors::InvalidArgument(
                       "Input(X) of FrameOp should be a tensor which contains "
                       "at least 1 dimension, but got rank %s.",
                       x_rank));
    PADDLE_ENFORCE_GT(hop_length, 0,
                      platform::errors::InvalidArgument(
                          "Attribute(hop_length) of FrameOp should be greater "
                          "than 0, but got %s.",
                          hop_length));
    PADDLE_ENFORCE_EQ(
        (axis == 0 || axis == -1), true,
        platform::errors::InvalidArgument(
            "Attribute(axis) of FrameOp should 0 or -1, but got %s.", axis));

    std::vector<int64_t> output_shape;
    int seq_length;
    int n_frames;

    int start_axis;
    int end_axis;

    if (axis == 0) {
      seq_length = x_dims[0];
      start_axis = 1;
      end_axis = x_rank - 1;
    } else {
      seq_length = x_dims[x_rank - 1];
      start_axis = 0;
      end_axis = x_rank - 2;
    }

    bool contain_unknown_dim = phi::contain_unknown_dim(x_dims);
    bool check = ctx->IsRuntime() || !contain_unknown_dim;
    if (check) {
      PADDLE_ENFORCE_LE(frame_length, seq_length,
                        platform::errors::InvalidArgument(
                            "Attribute(frame_length) of FrameOp should be less "
                            "equal than sequence length, but got (%s) > (%s).",
                            frame_length, seq_length));
    }

    // It won't go into for loop when x_rank == 1U.
    for (int i = start_axis; i <= end_axis; i++) {
      output_shape.push_back(x_dims[i]);
    }

    if (seq_length == -1) {
      n_frames = -1;
    } else {
      n_frames = 1 + (seq_length - frame_length) / hop_length;
    }

    if (axis == 0) {
      // (n_frames, frame_length, ...)
      output_shape.insert(output_shape.begin(), frame_length);
      output_shape.insert(output_shape.begin(), n_frames);
    } else {
      // (..., frame_length, n_frames)
      output_shape.push_back(frame_length);
      output_shape.push_back(n_frames);
    }

    ctx->SetOutputDim("Out", phi::make_ddim(output_shape));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    const auto in_dtype = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(in_dtype, ctx.GetPlace());
  }
};

class FrameOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of frame op.");
    AddOutput("Out", "(Tensor), The output tensor of frame op.");
    AddAttr<int>(
        "frame_length",
        "Length of the frame and `0 < frame_length <= x.shape[axis]`.");
    AddAttr<int>("hop_length",
                 "Number of steps to advance between adjacent frames and "
                 "`0 < hop_length`.");
    AddAttr<int>("axis",
                 "Specify the axis to operate on the input Tensors. Its value "
                 "should be 0(the first dimension) or -1(the last dimension).")
        .SetDefault(-1);
    AddComment(R"DOC(
      Slice the N-dimensional (where N >= 1) input into (overlapping) frames.
    )DOC");
  }
};

class FrameOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "frame_grad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "frame_grad");
    const auto x_dims = ctx->GetInputDim("X");
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    const auto in_dtype = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(in_dtype, ctx.GetPlace());
  }
};

template <typename T>
class FrameOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("frame_grad");
    retv->SetInput("X", this->Input("X"));
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(frame, ops::FrameOp, ops::FrameOpMaker,
                  ops::FrameOpGradMaker<paddle::framework::OpDesc>,
                  ops::FrameOpGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(frame_grad, ops::FrameOpGrad);

REGISTER_OP_CPU_KERNEL(
    frame, ops::FrameKernel<paddle::platform::CPUDeviceContext, int>,
    ops::FrameKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::FrameKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FrameKernel<paddle::platform::CPUDeviceContext, double>,
    ops::FrameKernel<paddle::platform::CPUDeviceContext,
                     paddle::platform::complex<float>>,
    ops::FrameKernel<paddle::platform::CPUDeviceContext,
                     paddle::platform::complex<double>>);

REGISTER_OP_CPU_KERNEL(
    frame_grad, ops::FrameGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::FrameGradKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::FrameGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FrameGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::FrameGradKernel<paddle::platform::CPUDeviceContext,
                         paddle::platform::complex<float>>,
    ops::FrameGradKernel<paddle::platform::CPUDeviceContext,
                         paddle::platform::complex<double>>);
