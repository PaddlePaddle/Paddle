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

// #include "paddle/fluid/operators/fused/acc_merge_op.h"

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace operators {
using Tensor = phi::DenseTensor;

class AccMergeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Acc"), "Input", "Acc", "AccMergeOp");
    OP_INOUT_CHECK(ctx->HasInput("Total"), "Input", "Total", "AccMergeOp");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "AccMergeOp");
    OP_INOUT_CHECK(ctx->HasOutput("Step"), "Output", "Step", "AccMergeOp");

    std::vector<int64_t> out_dims;
    out_dims.push_back({2});
    out_dims.push_back({2});

    ctx->SetOutputDim("Out", phi::make_ddim(out_dims));
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "Acc");
    return framework::OpKernelType(data_type, ctx.GetPlace(), layout, library);
  }
};

class AccMergeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Acc", "The input tensor Acc .");
    AddInput("Total", "The input tensor Total.");

    AddOutput("Out", "The output tensor Out.");
    AddOutput("Step", "The output Step.");

    AddComment(R"DOC(AccMerge Operator)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    acc_merge,
    ops::AccMergeOp,
    ops::AccMergeOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
