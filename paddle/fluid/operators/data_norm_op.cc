/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <memory>
#include <string>

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace operators {

using DataLayout = phi::DataLayout;

template <typename T>
using EigenArrayMap =
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using ConstEigenArrayMap =
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using EigenVectorArrayMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenVectorArrayMap =
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>;

class DataNormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "DataNorm");
    OP_INOUT_CHECK(
        ctx->HasInput("BatchSize"), "Input", "BatchSize", "DataNorm");
    OP_INOUT_CHECK(ctx->HasInput("BatchSum"), "Input", "BatchSum", "DataNorm");
    OP_INOUT_CHECK(
        ctx->HasInput("BatchSquareSum"), "Input", "BatchSquareSum", "DataNorm");
    OP_INOUT_CHECK(ctx->HasOutput("Means"), "Output", "Means", "DataNorm");
    OP_INOUT_CHECK(ctx->HasOutput("Scales"), "Output", "Scales", "DataNorm");
    OP_INOUT_CHECK(ctx->HasOutput("Y"), "Output", "Y", "DataNorm");
    bool enable_scale_and_shift =
        ctx->Attrs().Get<bool>("enable_scale_and_shift");
    if (enable_scale_and_shift) {
      PADDLE_ENFORCE_EQ(
          ctx->HasInput("scale_w"),
          true,
          common::errors::InvalidArgument(
              "Input(scale_w) of DataNormOp should not be null."));
      PADDLE_ENFORCE_EQ(ctx->HasInput("bias"),
                        true,
                        common::errors::InvalidArgument(
                            "Input(bias) of DataNormOp should not be null."));
    }

    const auto x_dims = ctx->GetInputDim("X");
    const DataLayout data_layout = common::StringToDataLayout(
        ctx->Attrs().Get<std::string>("data_layout"));

    PADDLE_ENFORCE_EQ(x_dims.size() >= 2 && x_dims.size() <= 5,
                      true,
                      common::errors::InvalidArgument(
                          "Input X must have 2 to 5 dimensions."));

    const int64_t C =
        (data_layout == DataLayout::kNCHW ? x_dims[1]
                                          : x_dims[x_dims.size() - 1]);

    PADDLE_ENFORCE_EQ(ctx->GetInputDim("BatchSize").size(),
                      1UL,
                      common::errors::InvalidArgument(
                          "The input dim of BatchSize should be 1"));
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("BatchSum").size(),
                      1UL,
                      common::errors::InvalidArgument(
                          "The input dim of BatchSum should be 1"));
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("BatchSquareSum").size(),
                      1UL,
                      common::errors::InvalidArgument(
                          "The input dim of BatchSquareSum should be 1"));
    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("BatchSize")[0],
                        C,
                        common::errors::InvalidArgument(
                            "The input dim[0] of BatchSize should be C"));
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("BatchSum")[0],
                        C,
                        common::errors::InvalidArgument(
                            "The input dim[0] of BatchSum should be C"));
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("BatchSquareSum")[0],
                        C,
                        common::errors::InvalidArgument(
                            "The input dim[0] of BatchSquareSum should be C"));
    }

    if (enable_scale_and_shift) {
      auto scale_dim = ctx->GetInputDim("scale_w");
      auto bias_dim = ctx->GetInputDim("bias");

      PADDLE_ENFORCE_EQ(
          scale_dim.size(),
          1UL,
          common::errors::InvalidArgument("the dimension of scale"
                                          "must equal to 1. But received: "
                                          "the shape of scale is [%s], "
                                          "the dimension of scale is [%d]",
                                          scale_dim,
                                          scale_dim.size()));
      PADDLE_ENFORCE_EQ(
          bias_dim.size(),
          1UL,
          common::errors::InvalidArgument("the dimension of bias"
                                          "must equal to 1. But received: "
                                          "the shape of bias is [%s],"
                                          "the dimension of bias is [%d]",
                                          bias_dim,
                                          bias_dim.size()));

      bool check = true;
      if ((!ctx->IsRuntime()) &&
          (common::product(scale_dim) <= 0 || common::product(bias_dim) <= 0)) {
        check = false;
      }

      if (check) {
        PADDLE_ENFORCE_EQ(scale_dim[0],
                          C,
                          common::errors::InvalidArgument(
                              "the shape of scale must equal to [%d]"
                              "But received: the shape of scale is [%d]",
                              C,
                              scale_dim[0]));
        PADDLE_ENFORCE_EQ(bias_dim[0],
                          C,
                          common::errors::InvalidArgument(
                              "the shape of bias must equal to [%d]"
                              "But received: the shape of bias is [%d]",
                              C,
                              bias_dim[0]));
      }
    }

    ctx->SetOutputDim("Y", x_dims);
    ctx->SetOutputDim("Means", {C});
    ctx->SetOutputDim("Scales", {C});
    ctx->ShareLoD("X", "Y");
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    // By default, the type of the scale, bias, mean,
    // and var tensors should both be float. (For float or float16 input tensor)
    // or double (For double input tensor).
    auto dn_param_type = framework::proto::VarType::FP32;
    if (input_data_type == framework::proto::VarType::FP64) {
      dn_param_type = framework::proto::VarType::FP64;
    }
    PADDLE_ENFORCE_EQ(dn_param_type,
                      OperatorWithKernel::IndicateVarDataType(ctx, "BatchSize"),
                      common::errors::InvalidArgument(
                          "BatchSize input should be of float type"));
    PADDLE_ENFORCE_EQ(dn_param_type,
                      OperatorWithKernel::IndicateVarDataType(ctx, "BatchSum"),
                      common::errors::InvalidArgument(
                          "BatchSum input should be of float type"));
    PADDLE_ENFORCE_EQ(
        dn_param_type,
        OperatorWithKernel::IndicateVarDataType(ctx, "BatchSquareSum"),
        common::errors::InvalidArgument(
            "BatchSquareSum input should be of float type"));

    bool enable_scale_and_shift = ctx.Attr<bool>("enable_scale_and_shift");
    if (enable_scale_and_shift) {
      PADDLE_ENFORCE_EQ(dn_param_type,
                        OperatorWithKernel::IndicateVarDataType(ctx, "scale_w"),
                        common::errors::InvalidArgument(
                            "scale_w input should be of float type"));
      PADDLE_ENFORCE_EQ(dn_param_type,
                        OperatorWithKernel::IndicateVarDataType(ctx, "bias"),
                        common::errors::InvalidArgument(
                            "bias input should be of float type"));
    }

    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }
};

class DataNormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    // AddAttr<bool>("is_test", "").SetDefault(false);
    AddAttr<float>("epsilon", "")
        .SetDefault(1e-4)
        .AddCustomChecker([](const float &epsilon) {
          PADDLE_ENFORCE_EQ(epsilon >= 0.0f && epsilon <= 0.001f,
                            true,
                            common::errors::InvalidArgument(
                                "'epsilon' should be between 0.0 and 0.001."));
        });
    AddAttr<int>("slot_dim",
                 "(int, default -1) Dimension of one slot if set, "
                 "when the input is concated by slot-wise embeddings")
        .SetDefault(-1);
    AddAttr<float>(
        "summary_decay_rate",
        "(float, default 0.9999999) The decay rate when update the summary")
        .SetDefault(0.9999999);
    AddAttr<bool>(
        "enable_scale_and_shift",
        "(bool, default false) Set to true to enable scale and shift such as "
        "batch_norm op")
        .SetDefault(false);
    AddInput("scale_w",
             "scale_w is a 1-dimensional tensor of size C "
             "that is applied to the output")
        .AsDispensable();
    AddInput("bias",
             "bias is a 1-dimensional tensor of size C "
             "that is applied to the output")
        .AsDispensable();
    AddAttr<std::string>("data_layout", "").SetDefault("NCHW");
    AddAttr<bool>("sync_stats", "(bool, default false) only used in multi-GPU")
        .SetDefault(false);
    AddInput("X", "The input tensor");
    AddInput("BatchSize",
             "BatchSize is a 1-dimensional tensor of size C "
             "that is applied to the output");
    AddInput("BatchSum",
             "BatchSum is a 1-dimensional tensor of size C "
             "that is applied to the output");
    AddInput("BatchSquareSum",
             "The global BatchSquareSum (for training) or "
             "estimated BatchSquareSum (for testing)");
    AddOutput("Y", "result after normalization");
    AddOutput("Means",
              "Mean of the history data batch, "
              "will apply to output when training")
        .AsIntermediate();
    AddOutput("Scales",
              "Scales of the history data batch, "
              "will apply to output when training")
        .AsIntermediate();
    AddComment(R"DOC(
Data Normalization.

Can be used as a normalizer function for data
The required data format for this layer is one of the following:
1. NHWC `[batch, in_height, in_width, in_channels]`
2. NCHW `[batch, in_channels, in_height, in_width]`

)DOC");
  }
};

class DataNormGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    // check input
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "DataNormGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Y")),
                   "Input",
                   framework::GradVarName("Y"),
                   "DataNormGrad");
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("BatchSize"),
        true,
        common::errors::NotFound(
            "Output(BatchSize) of DataNormGradOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("BatchSum"),
        true,
        common::errors::NotFound(
            "Output(BatchSum) of DataNormGradOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("BatchSquareSum"),
        true,
        common::errors::NotFound(
            "Output(BatchSquareSum) of DataNormGradOp should not be null."));
    OP_INOUT_CHECK(ctx->HasInput("Means"), "Input", "Means", "DataNormGrad");
    OP_INOUT_CHECK(ctx->HasInput("Scales"), "Input", "Scales", "DataNormGrad");
    bool enable_scale_and_shift =
        ctx->Attrs().Get<bool>("enable_scale_and_shift");
    // check output
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("BatchSize")),
                   "Output",
                   framework::GradVarName("BatchSize"),
                   "DataNormGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("BatchSum")),
                   "Output",
                   framework::GradVarName("BatchSum"),
                   "DataNormGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("BatchSquareSum")),
                   "Output",
                   framework::GradVarName("BatchSquareSum"),
                   "DataNormGrad");

    const auto x_dims = ctx->GetInputDim("X");
    const DataLayout data_layout = common::StringToDataLayout(
        ctx->Attrs().Get<std::string>("data_layout"));
    const int C = static_cast<int>(data_layout == DataLayout::kNCHW
                                       ? x_dims[1]
                                       : x_dims[x_dims.size() - 1]);

    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    }
    ctx->SetOutputDim(framework::GradVarName("BatchSize"), {C});
    ctx->SetOutputDim(framework::GradVarName("BatchSum"), {C});
    ctx->SetOutputDim(framework::GradVarName("BatchSquareSum"), {C});
    if (enable_scale_and_shift) {
      const bool has_scale_grad =
          ctx->HasOutput(framework::GradVarName("scale_w"));
      const bool has_bias_grad = ctx->HasOutput(framework::GradVarName("bias"));

      PADDLE_ENFORCE_EQ((has_scale_grad == has_bias_grad),
                        true,
                        common::errors::InvalidArgument(
                            "Output(Scale@GRAD) and Output(Bias@GRAD)"
                            "must be null or not be null at same time. "
                            "But now, has Scale@Grad=[%d], has Bias@GRAD=[%d]",
                            has_scale_grad,
                            has_bias_grad));
      if (has_scale_grad) {
        ctx->SetOutputDim(framework::GradVarName("scale_w"), {C});
        ctx->SetOutputDim(framework::GradVarName("bias"), {C});
      }
    }
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    const auto *var = ctx.InputVar(framework::GradVarName("Y"));
    if (var == nullptr) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Y@GRAD can not be found for computation"));
    }
    const phi::DenseTensor *t = nullptr;
    if (var->IsType<phi::DenseTensor>()) {
      t = &var->Get<phi::DenseTensor>();
    }
    if (t == nullptr) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Y@GRAD can not be found for computation"));
    }

    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return phi::KernelKey(data_type, ctx.GetPlace());
  }
};

template <typename T>
class DataNormGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("data_norm_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));

    op->SetInput("scale_w", this->Input("scale_w"));
    op->SetInput("bias", this->Input("bias"));
    op->SetOutput("BatchSize", this->Input("BatchSize"));
    op->SetOutput("BatchSum", this->Input("BatchSum"));
    op->SetOutput("BatchSquareSum", this->Input("BatchSquareSum"));
    op->SetInput("Scales", this->Output("Scales"));
    op->SetInput("Means", this->Output("Means"));

    op->SetAttrMap(this->Attrs());

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("BatchSize"),
                  this->InputGrad("BatchSize"));
    op->SetOutput(framework::GradVarName("BatchSum"),
                  this->InputGrad("BatchSum"));
    op->SetOutput(framework::GradVarName("BatchSquareSum"),
                  this->InputGrad("BatchSquareSum"));
    op->SetOutput(framework::GradVarName("scale_w"),
                  this->InputGrad("scale_w"));
    op->SetOutput(framework::GradVarName("bias"), this->InputGrad("bias"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(data_norm,
                  ops::DataNormOp,
                  ops::DataNormOpMaker,
                  ops::DataNormGradMaker<paddle::framework::OpDesc>,
                  ops::DataNormGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(data_norm_grad, ops::DataNormGradOp);

REGISTER_OP_VERSION(data_norm).AddCheckpoint(
    R"ROC(
              upgrade data_norm op by adding scale_w to support scale and shift.)ROC",
    paddle::framework::compatible::OpVersionDesc().NewInput(
        "scale_w",
        "scale_w is used to do scale during data_norm like batchnorm "));
