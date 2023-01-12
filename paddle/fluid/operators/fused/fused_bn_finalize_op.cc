/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"

namespace paddle {
namespace operators {

class FusedBnFinalizeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    // inputs
    AddInput("Sum", "");
    AddInput("SqSum", "");
    AddInput("Scale", "");
    AddInput("Bias", "");
    AddInput("inputRunningMean", "");
    AddInput("inputRunningVar", "");
    // outputs
    AddOutput("updatedRunningMean", "");
    AddOutput("updatedRunningVar", "");
    AddOutput("SavedMean", "");
    AddOutput("SavedInvVar", "");
    AddOutput("eqScale", "half");
    AddOutput("eqBias", "half");
    AddAttr<int64_t>("accumulation_count", "").SetDefault(1L);
    AddAttr<float>("momentum", "").SetDefault(0.9);
    AddAttr<float>("epsilon", "").SetDefault(1e-5);
    AddComment(R"DOC(
FusedBnFinalize Operator
Ref: https://github.com/NVIDIA/cudnn-frontend/blob/81a041a68245cd8f871c43bbbbd5b6b627979a30/samples/test_list.cpp#L1688

Conv + BN = Conv + Genstats + BN_Finalize

Requirements:
- All tensors should have shape [C], where C is the channel
  dimension of convolution input. Dtype should be FP32, except that
  eqScale and eqBias should be FP16.
)DOC");
  }
};

class FusedBnFinalizeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    // check basic IOs
    OP_INOUT_CHECK(ctx->HasInput("Sum"), "Input", "Sum", "FusedBnFinalize");
    OP_INOUT_CHECK(ctx->HasInput("SqSum"), "Input", "SqSum", "FusedBnFinalize");
    OP_INOUT_CHECK(ctx->HasInput("Scale"), "Input", "Scale", "FusedBnFinalize");
    OP_INOUT_CHECK(ctx->HasInput("Bias"), "Input", "Bias", "FusedBnFinalize");
    OP_INOUT_CHECK(ctx->HasInput("inputRunningMean"),
                   "Input",
                   "inputRunningMean",
                   "FusedBnFinalize");
    OP_INOUT_CHECK(ctx->HasInput("inputRunningVar"),
                   "Input",
                   "inputRunningVar",
                   "FusedBnFinalize");
    OP_INOUT_CHECK(ctx->HasOutput("updatedRunningMean"),
                   "Output",
                   "updatedRunningMean",
                   "FusedBnFinalize");
    OP_INOUT_CHECK(ctx->HasOutput("updatedRunningVar"),
                   "Output",
                   "updatedRunningVar",
                   "FusedBnFinalize");
    OP_INOUT_CHECK(
        ctx->HasOutput("SavedMean"), "Output", "SavedMean", "FusedBnFinalize");
    OP_INOUT_CHECK(ctx->HasOutput("SavedInvVar"),
                   "Output",
                   "SavedInvVar",
                   "FusedBnFinalize");
    OP_INOUT_CHECK(
        ctx->HasOutput("eqScale"), "Output", "eqScale", "FusedBnFinalize");
    OP_INOUT_CHECK(
        ctx->HasOutput("eqBias"), "Output", "eqBias", "FusedBnFinalize");
    OP_INOUT_CHECK(ctx->HasAttr("accumulation_count"),
                   "Attr",
                   "accumulation_count",
                   "FusedBnFinalize");

    auto c_dims = ctx->GetInputDim("Sum");  // [N, H, W, C]
    int64_t accumulation_count =
        ctx->Attrs().Get<int64_t>("accumulation_count");

    // sanity checks
    PADDLE_ENFORCE_EQ(
        (c_dims.size() == 1),
        1,
        platform::errors::InvalidArgument("Sum should be of shape [C]."
                                          "Got: [%s]",
                                          c_dims));

    PADDLE_ENFORCE_GT(
        accumulation_count,
        0,
        platform::errors::InvalidArgument(
            "Expect accumulation_count > 0, got %d", accumulation_count));

    // set output shapes
    ctx->SetOutputDim("updatedRunningMean", c_dims);
    ctx->SetOutputDim("updatedRunningVar", c_dims);
    ctx->SetOutputDim("SavedMean", c_dims);
    ctx->SetOutputDim("SavedInvVar", c_dims);
    ctx->SetOutputDim("eqScale", c_dims);
    ctx->SetOutputDim("eqBias", c_dims);
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "Sum");
    return framework::OpKernelType(data_type, ctx.GetPlace(), layout, library);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(fused_bn_finalize,
                             ops::FusedBnFinalizeOp,
                             ops::FusedBnFinalizeOpMaker);
