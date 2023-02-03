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
#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"

namespace paddle {
namespace operators {

class FusedScaleBiasAddReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    // inputs
    AddInput("X1", "");
    AddInput("Scale1", "");
    AddInput("Bias1", "");
    AddInput("X2", "");
    AddInput("Scale2", "").AsDispensable();
    AddInput("Bias2", "").AsDispensable();
    AddOutput("Y", "");

    AddAttr<bool>(
        "fuse_dual",
        R"DOC((bool, default false). Whether two resblocks are added before relu.)DOC")
        .SetDefault(false);
    AddComment(R"DOC(
FusedScaleBiasAddRelu Operator
Y = X1 * Scale1 + Bias1 + X2 (fuse_dual == false)
Y = X1 * Scale1 + Bias1 + X2 * Scale2 + Bias2 (fuse_dual == true)

Requirements:
- X should be FP16 and have NHWC layout
- Scale, Bias should have shape [C], where C is the channel
  dimension of X. Dtype should be FP32.
)DOC");
  }
};

class FusedScaleBiasAddReluOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    // check basic IOs
    OP_INOUT_CHECK(ctx->HasInput("X1"), "Input", "X1", "FusedScaleBiasAddRelu");
    OP_INOUT_CHECK(
        ctx->HasInput("Scale1"), "Input", "Scale1", "FusedScaleBiasAddRelu");
    OP_INOUT_CHECK(
        ctx->HasInput("Bias1"), "Input", "Bias1", "FusedScaleBiasAddRelu");
    OP_INOUT_CHECK(ctx->HasInput("X2"), "Input", "X2", "FusedScaleBiasAddRelu");

    bool fuse_dual = ctx->Attrs().Get<bool>("fuse_dual");
    if (fuse_dual) {
      OP_INOUT_CHECK(
          ctx->HasInput("Scale2"), "Input", "Scale2", "FusedScaleBiasAddRelu");
      OP_INOUT_CHECK(
          ctx->HasInput("Bias2"), "Input", "Bias2", "FusedScaleBiasAddRelu");
    }

    OP_INOUT_CHECK(ctx->HasOutput("Y"), "Output", "Y", "FusedScaleBiasAddRelu");

    auto x_dims = ctx->GetInputDim("X1");  // [N, H, W, C]

    // set output shapes
    ctx->SetOutputDim("Y", x_dims);
  }

  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X1");
    return phi::KernelKey(data_type, ctx.GetPlace());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(fused_scale_bias_add_relu,
                             ops::FusedScaleBiasAddReluOp,
                             ops::FusedScaleBiasAddReluOpMaker);
