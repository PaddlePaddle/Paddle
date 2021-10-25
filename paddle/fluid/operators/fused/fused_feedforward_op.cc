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

#include <algorithm>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/matmul_v2_op.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

class FusedFeedForwardOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "fused_feedforward");
    OP_INOUT_CHECK(context->HasInput("Linear1Weight"), "Input", "Linear1Weight",
                   "fused_feedforward");
    OP_INOUT_CHECK(context->HasInput("Linear2Weight"), "Input", "Linear2Weight",
                   "fused_feedforward");
    OP_INOUT_CHECK(context->HasOutput("Out"), "Output", "Out",
                   "fused_feedforward");
    OP_INOUT_CHECK(context->HasOutput("Dropout1Mask"), "Output", "Dropout1Mask",
                   "fused_feedforward");
    OP_INOUT_CHECK(context->HasOutput("Dropout2Mask"), "Output", "Dropout2Mask",
                   "fused_feedforward");
    OP_INOUT_CHECK(context->HasOutput("Ln1Mean"), "Output", "Ln1Mean",
                   "fused_feedforward");
    OP_INOUT_CHECK(context->HasOutput("Ln1Variance"), "Output", "Ln1Variance",
                   "fused_feedforward");
    OP_INOUT_CHECK(context->HasOutput("Ln2Mean"), "Output", "Ln2Mean",
                   "fused_feedforward");
    OP_INOUT_CHECK(context->HasOutput("Ln2Variance"), "Output", "Ln2Variance",
                   "fused_feedforward");
    OP_INOUT_CHECK(context->HasOutput("Linear1Out"), "Output", "Linear1Out",
                   "fused_feedforward");
    OP_INOUT_CHECK(context->HasOutput("Ln1Out"), "Output", "Ln1Out",
                   "fused_feedforward");
    OP_INOUT_CHECK(context->HasOutput("Dropout1Out"), "Output", "Dropout1Out",
                   "fused_feedforward");
    OP_INOUT_CHECK(context->HasOutput("Dropout2Out"), "Output", "Dropout2Out",
                   "fused_feedforward");

    auto dim_x = context->GetInputDim("X");
    auto mat_dim_x =
        math::CreateMatrixDescriptor(RowMatrixFromVector(dim_x), 0, false);
    // verify for the pre layer_norm, the feature size must be larger than 1
    PADDLE_ENFORCE_GT(
        mat_dim_x.width_, static_cast<size_t>(1),
        platform::errors::InvalidArgument("Product from the X shape[1] to "
                                          "shape[n-1] must be larger than 1!"));
    auto dim_Linear1Weight = context->GetInputDim("Linear1Weight");
    auto tmp_dim_x = dim_x;
    tmp_dim_x[dim_x.size() - 1] =
        dim_Linear1Weight[dim_Linear1Weight.size() - 1];
    context->SetOutputDim("Out", dim_x);
    if (context->Attrs().Get<bool>("dropout1_is_test") == false) {
      context->SetOutputDim("Dropout1Mask", tmp_dim_x);
    }
    context->SetOutputDim("Dropout1Out", tmp_dim_x);
    context->SetOutputDim("Linear1Out", tmp_dim_x);
    context->SetOutputDim("Ln1Out", dim_x);
    context->SetOutputDim("Dropout2Out", dim_x);

    if (context->Attrs().Get<bool>("dropout2_is_test") == false) {
      context->SetOutputDim("Dropout2Mask", dim_x);
    }
    framework::DDim mean_dim =
        framework::make_ddim({mat_dim_x.batch_size_ * mat_dim_x.height_});
    context->SetOutputDim("Ln1Mean", mean_dim);
    context->SetOutputDim("Ln1Variance", mean_dim);
    context->SetOutputDim("Ln2Mean", mean_dim);
    context->SetOutputDim("Ln2Variance", mean_dim);
    context->ShareLoD("X", "Out");
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class FusedFeedForwardOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input of FusedFeedForward op");
    AddInput(
        "Dropout1Seed",
        "The seed of first dropout op, it has higher priority than the attr "
        "fix_seed and seed")
        .AsDispensable();
    AddInput(
        "Dropout2Seed",
        "The seed of second dropout op, it has higher priority than the attr "
        "fix_seed and seed")
        .AsDispensable();

    AddInput("Linear1Weight", "The linear1 weight of FusedFeedForward op");
    AddInput("Linear1Bias", "The linear1 bias of FusedFeedForward op")
        .AsDispensable();
    AddInput("Linear2Weight", "The linear2 weight of FusedFeedForward op");
    AddInput("Linear2Bias", "The linear2 bias input of FusedFeedForward op")
        .AsDispensable();
    AddInput("Ln1Scale", "The layer_norm1 scale of FusedFeedForward op")
        .AsDispensable();
    AddInput("Ln1Bias", "The layer_norm1 bias of FusedFeedForward op")
        .AsDispensable();
    AddInput("Ln2Scale", "The layer_norm2 scale of FusedFeedForward op")
        .AsDispensable();
    AddInput("Ln2Bias", "The layer_norm2 bias of FusedFeedForward op")
        .AsDispensable();
    AddOutput("Out", "The output of FusedFeedForward op");
    AddOutput("Dropout1Mask", "The mask of dropout1").AsIntermediate();
    AddOutput("Dropout2Mask", "The mask of dropout2").AsIntermediate();
    AddOutput("Ln1Mean", "The mean of layer_norm1").AsIntermediate();
    AddOutput("Ln1Variance", "The variance of layer_norm1").AsIntermediate();
    AddOutput("Ln2Mean", "The mean of layer_nomr2").AsIntermediate();
    AddOutput("Ln2Variance", "The variance of layer_norm2").AsIntermediate();
    AddOutput("Linear1Out", "The output of linear1").AsIntermediate();
    AddOutput("Ln1Out", "The output of layer_norm1").AsIntermediate();
    AddOutput("Dropout1Out", "The output of dropout1").AsIntermediate();
    AddOutput("Dropout2Out", "The output of dropout2").AsIntermediate();

    AddAttr<bool>("pre_layer_norm", "true is pre layernorm").SetDefault(false);
    AddAttr<float>("ln1_epsilon", "epsilon of pre layer_norm")
        .SetDefault(1e-5f);
    AddAttr<float>("ln2_epsilon", "epsilon of post layer_norm")
        .SetDefault(1e-5f);
    AddAttr<std::string>("act_method", "act_method").SetDefault("gelu");
    AddAttr<float>("dropout1_rate", "the dropout rate of first dropout")
        .SetDefault(.5f)
        .AddCustomChecker([](const float &drop_p) {
          PADDLE_ENFORCE_EQ(
              drop_p >= 0.0f && drop_p <= 1.0f, true,
              platform::errors::InvalidArgument(
                  "'dropout1_rate' must be between 0.0 and 1.0."));
        });
    AddAttr<float>("dropout2_rate", "the dropout rate of second dropout")
        .SetDefault(.5f)
        .AddCustomChecker([](const float &drop_p) {
          PADDLE_ENFORCE_EQ(
              drop_p >= 0.0f && drop_p <= 1.0f, true,
              platform::errors::InvalidArgument(
                  "'dropout2_rate' must be between 0.0 and 1.0."));
        });
    AddAttr<std::string>("dropout1_implementation",
                         "the dropout implementation of first dropout")
        .SetDefault("downgrade_in_infer")
        .AddCustomChecker([](const std::string &type) {
          PADDLE_ENFORCE_EQ(
              type == "downgrade_in_infer" || type == "upscale_in_train", true,
              platform::errors::InvalidArgument(
                  "dropout1_implementation can only be downgrade_in_infer or "
                  "upscale_in_train"));
        });
    AddAttr<std::string>("dropout2_implementation",
                         "the dropout implementation of second dropout")
        .SetDefault("downgrade_in_infer")
        .AddCustomChecker([](const std::string &type) {
          PADDLE_ENFORCE_EQ(
              type == "downgrade_in_infer" || type == "upscale_in_train", true,
              platform::errors::InvalidArgument(
                  "dropout2_implementation can only be downgrade_in_infer or "
                  "upscale_in_train"));
        });
    AddAttr<bool>("dropout1_is_test", "the is_test of first dropout")
        .SetDefault(false);
    AddAttr<bool>("dropout2_is_test", "the is_test of second dropout")
        .SetDefault(false);
    AddAttr<bool>("dropout1_fix_seed", "the is_test of first dropout")
        .SetDefault(false);
    AddAttr<bool>("dropout2_fix_seed", "the is_test of second dropout")
        .SetDefault(false);
    AddAttr<int>("dropout1_seed", "Dropout1 random seed.").SetDefault(0);
    AddAttr<int>("dropout2_seed", "Dropout2 random seed.").SetDefault(0);
    AddComment(R"DOC(
        the function of fused_feedforward operator is the same as the following pseudo code:
        residual = src;
        ln1_out = src;
        if(pre_layer_norm){
            ln1_out = layer_norm(src);
        }
        out = linear(dropout(activation(dropout(linear(ln1_out)))));
        if(!pre_layer_norm) {
            out = layer_norm(out);
        }
        )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fused_feedforward, ops::FusedFeedForwardOp,
                  ops::FusedFeedForwardOpMaker);
