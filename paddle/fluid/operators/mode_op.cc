/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/mode_op.h"
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace operators {

class ModeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "mode");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "mode");
    OP_INOUT_CHECK(ctx->HasOutput("Indices"), "Output", "Indices", "mode");

    auto input_dims = ctx->GetInputDim("X");
    const int& dim_size = input_dims.size();
    int axis = static_cast<int>(ctx->Attrs().Get<int>("axis"));
    PADDLE_ENFORCE_EQ(
        (axis < dim_size) && (axis >= (-1 * dim_size)), true,
        paddle::platform::errors::InvalidArgument(
            "the axis of ModeOp must be [-%d, %d), but you set axis is %d",
            dim_size, dim_size, axis));
    PADDLE_ENFORCE_GE(input_dims.size(), 1,
                      paddle::platform::errors::InvalidArgument(
                          "input of ModeOp must have >= 1d shape"));
    if (axis < 0) axis += dim_size;
    bool keepdim = ctx->Attrs().Get<bool>("keepdim");
    std::vector<int64_t> dimvec;
    for (int64_t i = 0; i < axis; i++) {
      dimvec.emplace_back(input_dims[i]);
    }
    if (keepdim) {
      dimvec.emplace_back(static_cast<int64_t>(1));
    }
    for (int64_t i = axis + 1; i < dim_size; i++) {
      dimvec.emplace_back(input_dims[i]);
    }
    framework::DDim dims = framework::make_ddim(dimvec);
    PADDLE_ENFORCE_GE(input_dims.size(), 1, platform::errors::InvalidArgument(
                                                "input shape should >= 1d"));
    ctx->SetOutputDim("Out", dims);
    ctx->SetOutputDim("Indices", dims);
    ctx->ShareLoD("X", "Out");
    ctx->ShareLoD("X", "Indices");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    framework::LibraryType library_{framework::LibraryType::kPlain};
    framework::DataLayout layout_ = framework::DataLayout::kAnyLayout;
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.device_context(),
        layout_, library_);
  }
};

class ModeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) The input of Mode op");
    AddOutput("Out", "(Tensor) The output tensor of Topk op");
    AddOutput("Indices", "(Tensor) The indices of Topk elements of input");
    AddAttr<int>("axis",
                 "the axis to calculate mode values."
                 "if not set, will calculate on last axis.")
        .SetDefault(-1);
    AddAttr<bool>("keepdim", "Keep the dim that to reduce.").SetDefault(false);
    AddComment(R"DOC(
This operator finds the mode of input Tensor. And outputs their values and indices as vectors. 
)DOC");
  }
};

class ModeOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::InvalidArgument("Input(X) should be not null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Indices"), true,
        platform::errors::InvalidArgument("Input(Indices) should be not null"));
    PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Out")), true,
                      platform::errors::InvalidArgument(
                          "Grad Input(Out) should be not null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput(framework::GradVarName("X")), true,
        platform::errors::InvalidArgument("Grad Output(X) should be not null"));

    auto x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

template <typename T>
class ModeGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("mode_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput("X", this->Input("X"));
    op->SetInput("Indices", this->Output("Indices"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(mode, ops::ModeOp, ops::ModeOpMaker,
                  ops::ModeGradOpMaker<paddle::framework::OpDesc>,
                  ops::ModeGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(mode,
                       ops::ModeCPUKernel<paddle::platform::CPUPlace, float>,
                       ops::ModeCPUKernel<paddle::platform::CPUPlace, double>,
                       ops::ModeCPUKernel<paddle::platform::CPUPlace, int32_t>,
                       ops::ModeCPUKernel<paddle::platform::CPUPlace, int64_t>);

REGISTER_OPERATOR(mode_grad, ops::ModeOpGrad);
REGISTER_OP_CPU_KERNEL(
    mode_grad, ops::ModeGradCPUKernel<paddle::platform::CPUPlace, float>,
    ops::ModeGradCPUKernel<paddle::platform::CPUPlace, double>,
    ops::ModeGradCPUKernel<paddle::platform::CPUPlace, int32_t>,
    ops::ModeGradCPUKernel<paddle::platform::CPUPlace, int64_t>);
