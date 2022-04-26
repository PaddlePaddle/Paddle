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

#include "paddle/fluid/operators/overlap_add_op.h"

namespace paddle {
namespace operators {

class OverlapAddOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "overlap_add");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "overlap_add");

    const int hop_length = ctx->Attrs().Get<int>("hop_length");
    const int axis = ctx->Attrs().Get<int>("axis");

    const auto x_dims = ctx->GetInputDim("X");
    const int x_rank = x_dims.size();

    PADDLE_ENFORCE_GE(
        x_rank, 2,
        platform::errors::InvalidArgument(
            "Input(X) of OverlapAddOp should be a tensor which contains "
            "at least 2 dimensions, but got rank %s.",
            x_rank));

    PADDLE_ENFORCE_GT(
        hop_length, 0,
        platform::errors::InvalidArgument(
            "Attribute(hop_length) of OverlapAddOp should be greater "
            "than 0, but got %s.",
            hop_length));

    PADDLE_ENFORCE_EQ(
        (axis == 0 || axis == -1), true,
        platform::errors::InvalidArgument(
            "Attribute(axis) of OverlapAddOp should 0 or -1, but got %s.",
            axis));

    std::vector<int64_t> output_shape;
    int n_frames;
    int frame_length;
    int seq_length;

    int start_axis;
    int end_axis;
    if (axis == 0) {
      n_frames = x_dims[0];
      frame_length = x_dims[1];
      start_axis = 2;
      end_axis = x_rank - 1;
    } else {
      n_frames = x_dims[x_rank - 1];
      frame_length = x_dims[x_rank - 2];
      start_axis = 0;
      end_axis = x_rank - 3;
    }

    bool contain_unknown_dim = phi::contain_unknown_dim(x_dims);
    bool check = ctx->IsRuntime() || !contain_unknown_dim;
    if (check) {
      PADDLE_ENFORCE_LE(
          hop_length, frame_length,
          platform::errors::InvalidArgument(
              "Attribute(hop_length) of OverlapAddOp should be less or equal "
              "than frame_length, but got hop_length(%s) > frame_length(%s).",
              hop_length, frame_length));
    }

    if (n_frames == -1) {
      seq_length = -1;
    } else {
      seq_length = (n_frames - 1) * hop_length + frame_length;
    }

    // It won't go into for loop when x_rank == 2U.
    for (int i = start_axis; i <= end_axis; i++) {
      output_shape.push_back(x_dims[i]);
    }

    if (axis == 0) {
      // (seq_length, ...)
      output_shape.insert(output_shape.begin(), seq_length);
    } else {
      // (..., seq_length)
      output_shape.push_back(seq_length);
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

class OverlapAddOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of overlap_add op.");
    AddOutput("Out", "(Tensor), The output tensor of overlap_add op.");
    AddAttr<int>("hop_length",
                 "Number of steps to advance between adjacent frames and "
                 "`0 < hop_length <= frame_length`.");
    AddAttr<int>("axis",
                 "Specify the axis to operate on the input Tensors. Its value "
                 "should be 0(the first dimension) or -1(the last dimension).")
        .SetDefault(-1);
    AddComment(R"DOC(
      Reconstructs a tensor consisted of overlap added sequences from input frames.
    )DOC");
  }
};

class OverlapAddOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "overlap_add_grad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "overlap_add_grad");
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
class OverlapAddOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("overlap_add_grad");
    retv->SetInput("X", this->Input("X"));
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(overlap_add, ops::OverlapAddOp, ops::OverlapAddOpMaker,
                  ops::OverlapAddOpGradMaker<paddle::framework::OpDesc>,
                  ops::OverlapAddOpGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(overlap_add_grad, ops::OverlapAddOpGrad);

REGISTER_OP_CPU_KERNEL(
    overlap_add, ops::OverlapAddKernel<paddle::platform::CPUDeviceContext, int>,
    ops::OverlapAddKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::OverlapAddKernel<paddle::platform::CPUDeviceContext, float>,
    ops::OverlapAddKernel<paddle::platform::CPUDeviceContext, double>,
    ops::OverlapAddKernel<paddle::platform::CPUDeviceContext,
                          paddle::platform::complex<float>>,
    ops::OverlapAddKernel<paddle::platform::CPUDeviceContext,
                          paddle::platform::complex<double>>);

REGISTER_OP_CPU_KERNEL(
    overlap_add_grad,
    ops::OverlapAddGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::OverlapAddGradKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::OverlapAddGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::OverlapAddGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::OverlapAddGradKernel<paddle::platform::CPUDeviceContext,
                              paddle::platform::complex<float>>,
    ops::OverlapAddGradKernel<paddle::platform::CPUDeviceContext,
                              paddle::platform::complex<double>>);
