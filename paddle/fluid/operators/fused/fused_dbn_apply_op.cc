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

class FusedDbnApplyOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    // inputs
    AddInput("dY", "");
    AddInput("X", "");
    AddInput("A", "");
    AddInput("B", "");
    AddInput("C", "");
    // optional inputs for dual dBNApply
    AddInput("X_dual", "").AsDispensable();
    AddInput("A_dual", "").AsDispensable();
    AddInput("B_dual", "").AsDispensable();
    AddInput("C_dual", "").AsDispensable();
    // outputs
    AddOutput("dX", "");
    // optional outputs for dual dBNApply
    AddOutput("dX_dual", "").AsDispensable();
    // attributes
    AddAttr<bool>("fuse_dual", R"DOC((bool, default false))DOC")
        .SetDefault(false);
    AddComment(R"DOC(
FusedDbnApplyOpMaker Operator
Ref: https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#dualdbnapply
By default it performs the following:
dX = A* dY + B * X + C
With fuse_dual:
dX = A * dY + B * X + C
dX_dual = A_dual * dY + B_dual * X_dual + C_dual

Requirements:
- dY, X, dX should have NHWC layout and FP16 dtype.
- A, B, C should have shape [c], where c is the channel
  dimension of X. The dtype should be FP32.
)DOC");
  }
};

class FusedDbnApplyOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    // check basic IOs
    OP_INOUT_CHECK(ctx->HasInput("dY"), "Input", "dY", "FusedDbnApply");
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "FusedDbnApply");
    OP_INOUT_CHECK(ctx->HasInput("A"), "Input", "A", "FusedDbnApply");
    OP_INOUT_CHECK(ctx->HasInput("B"), "Input", "B", "FusedDbnApply");
    OP_INOUT_CHECK(ctx->HasInput("C"), "Input", "C", "FusedDbnApply");
    OP_INOUT_CHECK(ctx->HasOutput("dX"), "Output", "dX", "FusedDbnApply");
    bool fuse_dual = ctx->Attrs().Get<bool>("fuse_dual");
    if (fuse_dual) {
      OP_INOUT_CHECK(
          ctx->HasInput("X_dual"), "Input", "X_dual", "FusedDbnApply");
      OP_INOUT_CHECK(
          ctx->HasInput("A_dual"), "Input", "A_dual", "FusedDbnApply");
      OP_INOUT_CHECK(
          ctx->HasInput("B_dual"), "Input", "B_dual", "FusedDbnApply");
      OP_INOUT_CHECK(
          ctx->HasInput("C_dual"), "Input", "C_dual", "FusedDbnApply");
      OP_INOUT_CHECK(
          ctx->HasOutput("dX_dual"), "Output", "dX_dual", "FusedDbnApply");
    }

    auto x_dims = ctx->GetInputDim("X");  // [N, H, W, C]
    auto c_dims = ctx->GetInputDim("A");  // [C]

    // sanity checks
    PADDLE_ENFORCE_EQ(
        (c_dims.size() == 1) && (c_dims[0] == x_dims[x_dims.size() - 1]),
        1,
        platform::errors::InvalidArgument("A should be of shape [%d]."
                                          "Got: [%s]",
                                          x_dims[x_dims.size() - 1],
                                          c_dims));

    PADDLE_ENFORCE_EQ(x_dims[x_dims.size() - 1] % 16,
                      0,
                      platform::errors::InvalidArgument(
                          "The channel number of X should be multiple of 16."
                          "Got: %s",
                          x_dims[x_dims.size() - 1]));
    // set output shapes
    ctx->SetOutputDim("dX", x_dims);
    if (fuse_dual) {
      ctx->SetOutputDim("dX_dual", x_dims);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "dY");
    return framework::OpKernelType(data_type, ctx.GetPlace(), layout, library);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(fused_dbn_apply,
                             ops::FusedDbnApplyOp,
                             ops::FusedDbnApplyOpMaker);
