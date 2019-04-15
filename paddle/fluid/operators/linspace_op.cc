/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/linspace_op.h"

namespace paddle {
namespace operators {

class LinspaceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Start"),
                   "Input(Start) of LinspaceOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Stop"),
                   "Input(Stop) of LinspaceOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Num"),
                   "Input(Num) of LinspaceOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(OUt) of LinspaceOp should not be null.");

    auto s_dims = ctx->GetInputDim("Start");
    PADDLE_ENFORCE((s_dims.size() == 1) && (s_dims[0] == 1),
                   "The shape of Input(Start) should be [1].");

    auto e_dims = ctx->GetInputDim("Stop");
    PADDLE_ENFORCE((e_dims.size() == 1) && (e_dims[0] == 1),
                   "The shape of Input(Stop) should be [1].");

    auto step_dims = ctx->GetInputDim("Num");
    PADDLE_ENFORCE((step_dims.size() == 1) && (step_dims[0] == 1),
                   "The shape of Input(Num) should be [1].");

    ctx->SetOutputDim("Out", {-1});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    framework::LibraryType library_{framework::LibraryType::kPlain};
    framework::DataLayout layout_ = framework::DataLayout::kAnyLayout;
    return framework::OpKernelType(
        ctx.Input<framework::Tensor>("Start")->type(), ctx.device_context(),
        layout_, library_);
  }
};

class LinspaceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Start",
             "First entry in the sequence. It is a tensor of shape [1], should "
             "be of type float32 or float64.");
    AddInput("Stop",
             "Last entry in the sequence. It is a tensor of shape [1], should "
             "be of type float32 or float64.");
    AddInput("Num",
             "Number of entry in the sequence. It is a tensor of shape [1], "
             "should be of type int32.");
    AddOutput("Out", "A sequence of numbers.");
    AddComment(R"DOC(
    Return fixed number of evenly spaced values within a given interval. First entry is start, and last entry is stop. In the case when Num is 1, only Start is returned. Like linspace function of numpy.
)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(linspace, ops::LinspaceOp, ops::LinspaceOpMaker);
REGISTER_OP_CPU_KERNEL(linspace, ops::CPULinspaceKernel<float>,
                       ops::CPULinspaceKernel<double>);
