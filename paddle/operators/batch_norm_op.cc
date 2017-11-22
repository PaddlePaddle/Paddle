/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/batch_norm_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

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

class BatchNormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of BatchNormOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Scale"),
                   "Input(Scale) of BatchNormOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Bias"),
                   "Input(Bias) of BatchNormOp should not be null");
    PADDLE_ENFORCE(ctx->HasInput("Mean"),
                   "Input(Mean) of BatchNormOp should not be null");
    PADDLE_ENFORCE(ctx->HasInput("Variance"),
                   "Input(Variance) of BatchNormOp should not be null");
    PADDLE_ENFORCE(ctx->HasOutput("Y"),
                   "Output(Y) of BatchNormOp should not be null");
    auto is_test = ctx->Attrs().Get<bool>("is_test");
    PADDLE_ENFORCE_NE(ctx->HasOutput("MeanOut"), is_test,
                      "Output(MeanOut) should be set only in trainning(i.e. "
                      "is_test=false) mode");
    PADDLE_ENFORCE_NE(
        ctx->HasOutput("VarianceOut"), is_test,
        "Output(VarianceOut) should be set only in trainning(i.e. "
        "is_test=false) mode");
    PADDLE_ENFORCE_NE(ctx->HasOutput("SavedMean"), is_test,
                      "Output(SavedMean) should be set only in trainning(i.e. "
                      "is_test=false) mode");
    PADDLE_ENFORCE_NE(
        ctx->HasOutput("SavedVariance"), is_test,
        "Output(SavedVariance) should be set only in trainning(i.e. "
        "is_test=false) mode");

    const float epsilon = ctx->Attrs().Get<float>("epsilon");
    PADDLE_ENFORCE_GE(epsilon, 0.0, "epsilon should be larger than 0");
    PADDLE_ENFORCE_LE(epsilon, 0.001, "epsilon should not be too large");

    if (!is_test) {
      // make sure Mean/MeanOut and Variance/VarianceOut share memory in Python
      PADDLE_ENFORCE_EQ(ctx->Inputs("Mean")[0], ctx->Outputs("MeanOut")[0],
                        "Mean and MeanOut should share the same memory");
      PADDLE_ENFORCE_EQ(
          ctx->Inputs("Variance")[0], ctx->Outputs("VarianceOut")[0],
          "Variance and VarianceOut should share the same memory");
    }

    const auto x_dims = ctx->GetInputDim("X");
    const TensorFormat tensor_format =
        StringToTensorFormat(ctx->Attrs().Get<std::string>("tensor_format"));
    const int C = (tensor_format == TensorFormat::NCHW ||
                           tensor_format == TensorFormat::NCDHW
                       ? x_dims[1]
                       : x_dims[x_dims.size() - 1]);
    ValidTensorRank(x_dims.size(), tensor_format);

    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Scale").size(), 1UL);
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Scale")[0], C);
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Bias").size(), 1UL);
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Bias")[0], C);

    ctx->SetOutputDim("Y", x_dims);
    if (!is_test) {
      ctx->SetOutputDim("MeanOut", {C});
      ctx->SetOutputDim("VarianceOut", {C});
      ctx->SetOutputDim("SavedMean", {C});
      ctx->SetOutputDim("SavedVariance", {C});
    }
  }
};

class BatchNormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  BatchNormOpMaker(framework::OpProto *proto,
                   framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddAttr<bool>("is_test", "(bool, default false) Mode of operator.")
        .SetDefault(false);
    AddAttr<bool>("unbiased_estimate",
                  "(bool, default true) Whether use Bessel's correction when "
                  "estimating batch variance.")
        .SetDefault(true);
    AddAttr<float>("momentum",
                   "(float, default 0.9) Momentum for the moving average.")
        .SetDefault(0.9);
    AddAttr<float>("epsilon",
                   "(float, default 1e-5) A small float number added to "
                   "variance to prevent numerical error.")
        .SetDefault(static_cast<float>(1e-5));
    AddAttr<std::string>("tensor_format",
                         "(string, default NCHW) Layout of input tensor, one "
                         "of {NC, NHWC, NCHW, NDHWC, HCDHW}")
        .SetDefault("NCHW");
    AddInput("X", "The input tensor");
    AddInput("Scale",
             "Scale is a 1-dimensional tensor of size C "
             "that is applied to the output");
    AddInput("Bias",
             "Bias is a 1-dimensional tensor of size C "
             "that is applied to the output");
    AddInput("Mean",
             "The global mean (for training) or "
             "estimated mean (for testing)");
    AddInput("Variance",
             "The global variance (for training) "
             "or estimated Variance (for testing)");
    AddOutput("Y", "result after normalization");
    AddOutput("MeanOut",
              "Share memory with Mean. "
              "Store the global mean when training");
    AddOutput("VarianceOut",
              "Share memory with Variance. "
              "Store the global Variance when training");
    AddOutput("SavedMean",
              "Mean of the current mini batch, "
              "will apply to output when training")
        .AsIntermediate();
    AddOutput("SavedVariance",
              "Variance of the current mini batch, "
              "will apply to output when training")
        .AsIntermediate();
    AddComment(R"DOC(
Batch Normalization.

Batch Norm has been implemented as discussed in the paper:
https://arxiv.org/pdf/1502.03167.pdf
Can be used as a normalizer function for conv2d and fully_connected operations.
The required data format for this layer is one of the following:
For 2D tensor:
- NC `[batch, in_channels]`
For 4D tensor:
- NHWC `[batch, in_height, in_width, in_channels]`
- NCHW `[batch, in_channels, in_height, in_width]`
For 5D tensor:
- NCDHW `[batch, in_channels, in_depth, in_height, in_width]`
- NDHWC `[batch, in_depth, in_height, in_width, in_channels]`
)DOC");
  }
};

template <typename T>
class BatchNormKernel<platform::CPUPlace, T> : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const bool is_test = ctx.Attr<bool>("is_test");
    auto *y = ctx.Output<Tensor>("Y");
    // alloc memory
    y->mutable_data<T>(ctx.GetPlace());

    Tensor *mean_out = nullptr, *variance_out = nullptr, *saved_mean = nullptr,
           *saved_variance = nullptr;
    if (!is_test) {
      mean_out = ctx.Output<Tensor>("MeanOut");
      variance_out = ctx.Output<Tensor>("VarianceOut");
      saved_mean = ctx.Output<Tensor>("SavedMean");
      saved_variance = ctx.Output<Tensor>("SavedVariance");

      mean_out->mutable_data<T>(ctx.GetPlace());
      variance_out->mutable_data<T>(ctx.GetPlace());
      saved_mean->mutable_data<T>(ctx.GetPlace());
      saved_variance->mutable_data<T>(ctx.GetPlace());
    }

    BatchNormalizeForward<T>(
        ctx.GetPlace(), *ctx.Input<Tensor>("X"), *ctx.Input<Tensor>("Scale"),
        *ctx.Input<Tensor>("Bias"), *ctx.Input<Tensor>("Mean"),
        *ctx.Input<Tensor>("Variance"), ctx.Attr<float>("epsilon"),
        ctx.Attr<float>("momentum"), ctx.Attr<std::string>("tensor_format"),
        ctx.Attr<bool>("unbiased_estimate"), is_test, y, mean_out, variance_out,
        saved_mean, saved_variance);
  }
};

class BatchNormGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of BatchNormGradOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Scale"),
                   "Input(Scale) of BatchNormGradOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Y")),
                   "Input(Y@Grad) of BatchNormGradOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("SavedMean"),
                   "Input(SavedMean) of BatchNormGradOp should not be null");
    PADDLE_ENFORCE(
        ctx->HasInput("SavedVariance"),
        "Input(SavedVariance) of BatchNormGradOp should not be null");
    PADDLE_ENFORCE(!ctx->Attrs().Get<bool>("is_test"),
                   "Attribute(is_test) of BatchNormGradOp should be false, "
                   "i.e. gradient computation is not allowed in test mode.");
    // check output
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                   "Output(X@Grad) of BatchNormGradOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("Scale")),
                   "Output(Scale@Grad) of BatchNormGradOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("Bias")),
                   "Output(Bias@Grad) of BatchNormGradOp should not be null.");

    const auto x_dims = ctx->GetInputDim("X");
    const TensorFormat tensor_format =
        StringToTensorFormat(ctx->Attrs().Get<std::string>("tensor_format"));
    const int C = (tensor_format == TensorFormat::NCHW ||
                           tensor_format == TensorFormat::NCDHW
                       ? x_dims[1]
                       : x_dims[x_dims.size() - 1]);

    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    ctx->SetOutputDim(framework::GradVarName("Scale"), {C});
    ctx->SetOutputDim(framework::GradVarName("Bias"), {C});
  }

 protected:
  framework::OpKernelType GetKernelType(
      const framework::ExecutionContext &ctx) const override {
    const auto *var = ctx.InputVar(framework::GradVarName("Y"));
    if (var == nullptr) {
      PADDLE_THROW("can't find Y@GRAD");
    }
    const Tensor *t = nullptr;
    if (var->IsType<Tensor>()) {
      t = &var->Get<Tensor>();
    } else if (var->IsType<LoDTensor>()) {
      t = &var->Get<LoDTensor>();
    }
    if (t == nullptr) {
      PADDLE_THROW("can't find Y@GRAD");
    }
    return framework::OpKernelType(framework::ToDataType(t->type()),
                                   ctx.device_context());
  }
};

template <typename T>
class BatchNormGradKernel<platform::CPUPlace, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    // init output
    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));
    d_x->mutable_data<T>(ctx.GetPlace());
    d_scale->mutable_data<T>(ctx.GetPlace());
    d_bias->mutable_data<T>(ctx.GetPlace());

    BatchNormalizeBackward<T>(
        ctx.GetPlace(), *ctx.Input<Tensor>(framework::GradVarName("Y")),
        *ctx.Input<Tensor>("X"), *ctx.Input<Tensor>("Scale"),
        *ctx.Input<Tensor>("SavedMean"), *ctx.Input<Tensor>("SavedVariance"),
        ctx.Attr<std::string>("tensor_format"), d_x, d_scale, d_bias);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(batch_norm, ops::BatchNormOp, ops::BatchNormOpMaker,
            batch_norm_grad, ops::BatchNormGradOp);
REGISTER_OP_CPU_KERNEL(batch_norm,
                       ops::BatchNormKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    batch_norm_grad,
    ops::BatchNormGradKernel<paddle::platform::CPUPlace, float>);
