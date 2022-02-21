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
    AddInput("X", "");
    AddOutput("Out", "");
    AddAttr<int>("n_fft", "");
    AddAttr<int>("hop_length", "");
    AddAttr<bool>("normalized", "");
    AddAttr<bool>("onesided", "");
    AddComment(R"DOC(
      Stft Op.
    )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(stft, ops::StftOp, ops::StftOpMaker);
REGISTER_OP_CPU_KERNEL(
    stft, ops::StftKernel<paddle::platform::CPUDeviceContext, float>,
    ops::StftKernel<paddle::platform::CPUDeviceContext, double>);
