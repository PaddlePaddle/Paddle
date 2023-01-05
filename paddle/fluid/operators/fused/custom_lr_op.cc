/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2022 NVIDIA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// #include "paddle/fluid/operators/fused/custom_lr_op.h"

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace operators {
using Tensor = phi::DenseTensor;

class CustomLrOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "CustomLrOp");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "CustomLrOp");

    // auto base_lr = ctx->Attrs().Get<float>("base_lr");
    // auto max_step = ctx->Attrs().Get<int64_t>("max_step");

    auto out_dims = ctx->GetInputDim("X");;

    ctx->SetOutputDim("Out", out_dims);
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.GetPlace(), layout, library);
  }
};

class CustomLrOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor X.");

    AddOutput("Out", "The output tensor Out.");

    AddAttr<float>(
        "base_lr",
        R"DOC((float, default 0.0f)DOC")
        .SetDefault(0.0f);

    AddAttr<int64_t>(
        "max_step",
        R"DOC((int64_t, default 0))DOC")
        .SetDefault(0);

    AddComment(R"DOC(CustomLr Operator)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    custom_lr,
    ops::CustomLrOp,
    ops::CustomLrOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
