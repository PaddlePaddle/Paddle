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
    OP_INOUT_CHECK(context->HasOutput("Linear1Out"), "Output", "Linear1Out",
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
    context->SetOutputDim("Dropout2Out", dim_x);

    if (context->Attrs().Get<bool>("dropout2_is_test") == false) {
      context->SetOutputDim("Dropout2Mask", dim_x);
    }
    framework::DDim mean_dim =
        framework::make_ddim({mat_dim_x.batch_size_ * mat_dim_x.height_});
    bool pre_layer_norm = context->Attrs().Get<bool>("pre_layer_norm");
    if (pre_layer_norm) {
      OP_INOUT_CHECK(context->HasOutput("Ln1Mean"), "Output", "Ln1Mean",
                     "fused_feedforward");
      OP_INOUT_CHECK(context->HasOutput("Ln1Variance"), "Output", "Ln1Variance",
                     "fused_feedforward");
      OP_INOUT_CHECK(context->HasOutput("Ln1Out"), "Output", "Ln1Out",
                     "fused_feedforward");
      context->SetOutputDim("Ln1Out", dim_x);
      context->SetOutputDim("Ln1Mean", mean_dim);
      context->SetOutputDim("Ln1Variance", mean_dim);
    } else {
      OP_INOUT_CHECK(context->HasOutput("Ln2Mean"), "Output", "Ln2Mean",
                     "fused_feedforward");
      OP_INOUT_CHECK(context->HasOutput("Ln2Variance"), "Output", "Ln2Variance",
                     "fused_feedforward");
      context->SetOutputDim("Ln2Mean", mean_dim);
      context->SetOutputDim("Ln2Variance", mean_dim);
    }
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

class FusedFeedForwardOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->Attrs().Get<bool>("dropout1_is_test"), false,
                      platform::errors::InvalidArgument(
                          "GradOp is only callable when is_test is false"));
    PADDLE_ENFORCE_EQ(ctx->Attrs().Get<bool>("dropout2_is_test"), false,
                      platform::errors::InvalidArgument(
                          "GradOp is only callable when is_test is false"));
    bool pre_layer_norm = ctx->Attrs().Get<bool>("pre_layer_norm");
    OP_INOUT_CHECK(ctx->HasInput("Dropout1Mask"), "Input", "Dropout1Mask",
                   "FusedFeedForwardGrad");
    OP_INOUT_CHECK(ctx->HasInput("Dropout2Mask"), "Input", "Dropout1Mask",
                   "FusedFeedForwardGrad");
    OP_INOUT_CHECK(ctx->HasInput("Linear1Out"), "Input", "Linear1Out",
                   "FusedFeedForwardGrad");
    OP_INOUT_CHECK(ctx->HasInput("Dropout1Out"), "Input", "Dropout1Out",
                   "FusedFeedForwardGrad");
    OP_INOUT_CHECK(ctx->HasInput("Dropout2Out"), "Input", "Dropout2Out",
                   "FusedFeedForwardGrad");
    OP_INOUT_CHECK(ctx->HasInput("Linear1Weight"), "Input", "Linear1Weight",
                   "FusedFeedForwardGrad");
    OP_INOUT_CHECK(ctx->HasInput("Linear2Weight"), "Input", "Linear2Weight",
                   "FusedFeedForwardGrad");
    if (pre_layer_norm) {
      OP_INOUT_CHECK(ctx->HasInput("Ln1Mean"), "Input", "Ln1Mean",
                     "FusedFeedForwardGrad");
      OP_INOUT_CHECK(ctx->HasInput("Ln1Variance"), "Input", "Ln1Variance",
                     "FusedFeedForwardGrad");
      OP_INOUT_CHECK(ctx->HasInput("Ln1Out"), "Input", "Ln1Out",
                     "FusedFeedForwardGrad");
    } else {
      OP_INOUT_CHECK(ctx->HasInput("Ln2Mean"), "Input", "Ln2Mean",
                     "FusedFeedForwardGrad");
      OP_INOUT_CHECK(ctx->HasInput("Ln2Variance"), "Input", "Ln2Variance",
                     "FusedFeedForwardGrad");
    }

    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "FusedFeedForwardGrad");

    auto d_out_dim = ctx->GetInputDim(framework::GradVarName("Out"));
    ctx->SetOutputDim(framework::GradVarName("X"), d_out_dim);
    if (ctx->HasOutput(framework::GradVarName("Ln1Scale"))) {
      ctx->SetOutputDim(framework::GradVarName("Ln1Scale"),
                        ctx->GetInputDim("Ln1Scale"));
    }
    if (ctx->HasOutput(framework::GradVarName("Ln1Bias"))) {
      ctx->SetOutputDim(framework::GradVarName("Ln1Bias"),
                        ctx->GetInputDim("Ln1Bias"));
    }
    if (ctx->HasOutput(framework::GradVarName("Ln2Scale"))) {
      ctx->SetOutputDim(framework::GradVarName("Ln2Scale"),
                        ctx->GetInputDim("Ln2Scale"));
    }
    if (ctx->HasOutput(framework::GradVarName("Ln2Bias"))) {
      ctx->SetOutputDim(framework::GradVarName("Ln2Bias"),
                        ctx->GetInputDim("Ln2Bias"));
    }
    ctx->SetOutputDim(framework::GradVarName("Linear1Weight"),
                      ctx->GetInputDim("Linear1Weight"));
    if (ctx->HasOutput(framework::GradVarName("Linear1Bias"))) {
      ctx->SetOutputDim(framework::GradVarName("Linear1Bias"),
                        ctx->GetInputDim("Linear1Bias"));
    }
    ctx->SetOutputDim(framework::GradVarName("Linear2Weight"),
                      ctx->GetInputDim("Linear2Weight"));
    if (ctx->HasOutput(framework::GradVarName("Linear2Bias"))) {
      ctx->SetOutputDim(framework::GradVarName("Linear2Bias"),
                        ctx->GetInputDim("Linear2Bias"));
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input = ctx.Input<Tensor>("X");
    auto input_data_type = framework::TransToProtoVarType(input->dtype());
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

template <typename T>
class FusedFeedForwardOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("fused_feedforward_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput("X", this->Input("X"));
    op->SetInput("Linear1Weight", this->Input("Linear1Weight"));
    op->SetInput("Linear1Bias", this->Input("Linear1Bias"));
    op->SetInput("Linear2Weight", this->Input("Linear2Weight"));
    op->SetInput("Dropout1Mask", this->Output("Dropout1Mask"));
    op->SetInput("Dropout2Mask", this->Output("Dropout2Mask"));
    op->SetInput("Linear1Out", this->Output("Linear1Out"));
    op->SetInput("Dropout1Out", this->Output("Dropout1Out"));
    op->SetInput("Dropout2Out", this->Output("Dropout2Out"));

    op->SetAttrMap(this->Attrs());
    bool pre_layer_norm = BOOST_GET_CONST(bool, op->GetAttr("pre_layer_norm"));

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    if (pre_layer_norm) {
      op->SetInput("Ln1Scale", this->Input("Ln1Scale"));
      op->SetInput("Ln1Bias", this->Input("Ln1Bias"));
      op->SetInput("Ln1Out", this->Output("Ln1Out"));
      op->SetInput("Ln1Mean", this->Output("Ln1Mean"));
      op->SetInput("Ln1Variance", this->Output("Ln1Variance"));
      op->SetOutput(framework::GradVarName("Ln1Scale"),
                    this->InputGrad("Ln1Scale"));
      op->SetOutput(framework::GradVarName("Ln1Bias"),
                    this->InputGrad("Ln1Bias"));
    } else {
      op->SetInput("Ln2Scale", this->Input("Ln2Scale"));
      op->SetInput("Ln2Bias", this->Input("Ln2Bias"));
      op->SetInput("Ln2Mean", this->Output("Ln2Mean"));
      op->SetInput("Ln2Variance", this->Output("Ln2Variance"));
      op->SetOutput(framework::GradVarName("Ln2Scale"),
                    this->InputGrad("Ln2Scale"));
      op->SetOutput(framework::GradVarName("Ln2Bias"),
                    this->InputGrad("Ln2Bias"));
    }
    op->SetOutput(framework::GradVarName("Linear1Weight"),
                  this->InputGrad("Linear1Weight"));
    op->SetOutput(framework::GradVarName("Linear1Bias"),
                  this->InputGrad("Linear1Bias"));
    op->SetOutput(framework::GradVarName("Linear2Weight"),
                  this->InputGrad("Linear2Weight"));
    if (this->HasInput("Linear2Bias")) {
      op->SetInput("Linear2Bias", this->Input("Linear2Bias"));
      op->SetOutput(framework::GradVarName("Linear2Bias"),
                    this->InputGrad("Linear2Bias"));
    }
  }
};

template <typename T>
class FusedFeedForwardOpDoubleGradMaker
    : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {}
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fused_feedforward, ops::FusedFeedForwardOp,
                  ops::FusedFeedForwardOpMaker,
                  ops::FusedFeedForwardOpGradMaker<paddle::framework::OpDesc>,
                  ops::FusedFeedForwardOpGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(fused_feedforward_grad, ops::FusedFeedForwardOpGrad);
