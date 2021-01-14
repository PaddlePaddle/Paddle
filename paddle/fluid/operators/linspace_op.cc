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
#include <string>
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace operators {

class LinspaceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Start"), "Input", "Start", "linspace");
    OP_INOUT_CHECK(ctx->HasInput("Stop"), "Input", "Stop", "linspace");
    OP_INOUT_CHECK(ctx->HasInput("Num"), "Input", "Num", "linspace");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "linspace");

    auto s_dims = ctx->GetInputDim("Start");
    PADDLE_ENFORCE_EQ((s_dims.size() == 1) && (s_dims[0] == 1), true,
                      platform::errors::InvalidArgument(
                          "The shape of Input(Start) must be [1],"
                          "but received input shape is [%s].",
                          s_dims));
    auto e_dims = ctx->GetInputDim("Stop");
    PADDLE_ENFORCE_EQ((e_dims.size() == 1) && (e_dims[0] == 1), true,
                      platform::errors::InvalidArgument(
                          "The shape of Input(Stop) must be [1],"
                          "but received input shape is [%s].",
                          e_dims));
    auto step_dims = ctx->GetInputDim("Num");
    PADDLE_ENFORCE_EQ(
        (step_dims.size() == 1) && (step_dims[0] == 1), true,
        platform::errors::InvalidArgument("The shape of Input(Num) must be [1],"
                                          "but received input shape is [%s].",
                                          step_dims));
    ctx->SetOutputDim("Out", {-1});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::proto::VarType::Type(ctx.Attr<int>("dtype")),
        ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const framework::Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    return expected_kernel_type;
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
    AddAttr<int>("dtype", "The output data type.");
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
                       ops::CPULinspaceKernel<int32_t>,
                       ops::CPULinspaceKernel<int64_t>,
                       ops::CPULinspaceKernel<double>);

REGISTER_OP_VERSION(linspace)
    .AddCheckpoint(
        R"ROC(
      Upgrade linspace to add a new attribute [dtype].
    )ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "dtype", "In order to change output data type ", 5));
