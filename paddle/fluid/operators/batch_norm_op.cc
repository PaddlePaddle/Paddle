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

#include "paddle/fluid/operators/batch_norm_op.h"
#include <string>
#include "paddle/fluid/framework/data_layout.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DataLayout = framework::DataLayout;

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
    PADDLE_ENFORCE(ctx->HasInput("X"), "");
    PADDLE_ENFORCE(ctx->HasInput("Scale"), "");
    PADDLE_ENFORCE(ctx->HasInput("Bias"), "");
    PADDLE_ENFORCE(ctx->HasOutput("Y"), "");

    // make sure Mean/MeanOut and Variance/VarianceOut share memory in Python
    PADDLE_ENFORCE_EQ(ctx->Inputs("Mean")[0], ctx->Outputs("MeanOut")[0],
                      "Mean and MeanOut should share the same memory");
    PADDLE_ENFORCE_EQ(ctx->Inputs("Variance")[0],
                      ctx->Outputs("VarianceOut")[0],
                      "Variance and VarianceOut should share the same memory");

    const auto x_dims = ctx->GetInputDim("X");
    const DataLayout data_layout = framework::StringToDataLayout(
        ctx->Attrs().Get<std::string>("data_layout"));

    PADDLE_ENFORCE(x_dims.size() >= 2 && x_dims.size() <= 5,
                   "Input X must have 2 to 5 dimensions.");

    const int64_t C =
        (data_layout == DataLayout::kNCHW ? x_dims[1]
                                          : x_dims[x_dims.size() - 1]);
    // Only in test mode need to check the Input(Mean), Input(Variance)
    bool is_test = ctx->Attrs().Get<bool>("is_test");
    if (is_test) {
      PADDLE_ENFORCE(ctx->HasInput("Mean"), "");
      PADDLE_ENFORCE(ctx->HasInput("Variance"), "");
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("Mean"), 1UL);
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("Mean")[0], C);
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("Variance"), 1UL);
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("Variance")[0], C);
    } else {
      PADDLE_ENFORCE(ctx->HasOutput("MeanOut"), "");
      PADDLE_ENFORCE(ctx->HasOutput("VarianceOut"), "");
      PADDLE_ENFORCE(ctx->HasOutput("SavedMean"), "");
      PADDLE_ENFORCE(ctx->HasOutput("SavedVariance"), "");
      ctx->SetOutputDim("MeanOut", {C});
      ctx->SetOutputDim("VarianceOut", {C});
      ctx->SetOutputDim("SavedMean", {C});
      ctx->SetOutputDim("SavedVariance", {C});
    }

    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Scale").size(), 1UL);
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Scale")[0], C);
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Bias").size(), 1UL);
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Bias")[0], C);

    ctx->SetOutputDim("Y", x_dims);
    ctx->ShareLoD("X", "Y");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type =
        framework::ToDataType(ctx.Input<Tensor>("X")->type());
    // By default, the type of the scale, bias, mean,
    // and var tensors should both be float. (For float or float16 input tensor)
    // or double (For double input tensor).
    auto bn_param_type = framework::proto::VarType::FP32;
    if (input_data_type == framework::proto::VarType::FP64) {
      bn_param_type = framework::proto::VarType::FP64;
    }
    PADDLE_ENFORCE_EQ(bn_param_type,
                      framework::ToDataType(ctx.Input<Tensor>("Scale")->type()),
                      "Scale input should be of float type");
    PADDLE_ENFORCE_EQ(bn_param_type,
                      framework::ToDataType(ctx.Input<Tensor>("Bias")->type()),
                      "Bias input should be of float type");
    PADDLE_ENFORCE_EQ(bn_param_type,
                      framework::ToDataType(ctx.Input<Tensor>("Mean")->type()),
                      "Mean input should be of float type");
    PADDLE_ENFORCE_EQ(bn_param_type, framework::ToDataType(
                                         ctx.Input<Tensor>("Variance")->type()),
                      "Variance input should be of float type");

    framework::LibraryType library_{framework::LibraryType::kPlain};
#ifdef PADDLE_WITH_MKLDNN
    if (library_ == framework::LibraryType::kPlain &&
        platform::CanMKLDNNBeUsed(ctx)) {
      library_ = framework::LibraryType::kMKLDNN;
    }
#endif
    // TODO(pzelazko-intel): enable MKLDNN layout when it's ready
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    return framework::OpKernelType(input_data_type, ctx.GetPlace(), layout,
                                   library_);
  }
};

class BatchNormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  BatchNormOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddAttr<bool>("is_test", "").SetDefault(false);
    AddAttr<float>("momentum", "").SetDefault(0.9);
    AddAttr<float>("epsilon", "")
        .SetDefault(1e-5)
        .AddCustomChecker([](const float &epsilon) {
          PADDLE_ENFORCE(epsilon >= 0.0f && epsilon <= 0.001f,
                         "'epsilon' should be between 0.0 and 0.001.");
        });
    AddAttr<std::string>("data_layout", "").SetDefault("NCHW");
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
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
    AddComment(R"DOC(
Batch Normalization.

Batch Norm has been implemented as discussed in the paper:
https://arxiv.org/pdf/1502.03167.pdf
Can be used as a normalizer function for conv2d and fully_connected operations.
The required data format for this layer is one of the following:
1. NHWC `[batch, in_height, in_width, in_channels]`
2. NCHW `[batch, in_channels, in_height, in_width]`

)DOC");
  }
};

class BatchNormGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    // check input
    PADDLE_ENFORCE(ctx->HasInput("X"));
    PADDLE_ENFORCE(ctx->HasInput("Scale"), "");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Y")), "");
    PADDLE_ENFORCE(ctx->HasInput("SavedMean"), "");
    PADDLE_ENFORCE(ctx->HasInput("SavedVariance"), "");

    // check output
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")), "");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("Scale")), "");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("Bias")), "");

    const auto x_dims = ctx->GetInputDim("X");
    const DataLayout data_layout = framework::StringToDataLayout(
        ctx->Attrs().Get<std::string>("data_layout"));
    const int C =
        (data_layout == DataLayout::kNCHW ? x_dims[1]
                                          : x_dims[x_dims.size() - 1]);

    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    ctx->SetOutputDim(framework::GradVarName("Scale"), {C});
    ctx->SetOutputDim(framework::GradVarName("Bias"), {C});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
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

    framework::LibraryType library_{framework::LibraryType::kPlain};
#ifdef PADDLE_WITH_MKLDNN
    if (library_ == framework::LibraryType::kPlain &&
        platform::CanMKLDNNBeUsed(ctx)) {
      library_ = framework::LibraryType::kMKLDNN;
    }
#endif
    // TODO(pzelazko-intel): enable MKLDNN layout when it's ready
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("X")->type()), ctx.GetPlace(),
        layout, library_);
  }
};

class BatchNormGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto *op = new framework::OpDesc();
    op->SetType("batch_norm_grad");
    op->SetInput("X", Input("X"));
    op->SetInput(framework::GradVarName("Y"), OutputGrad("Y"));

    op->SetInput("Scale", Input("Scale"));
    op->SetInput("Bias", Input("Bias"));
    op->SetInput("SavedMean", Output("SavedMean"));
    op->SetInput("SavedVariance", Output("SavedVariance"));

    op->SetAttrMap(Attrs());

    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetOutput(framework::GradVarName("Scale"), InputGrad("Scale"));
    op->SetOutput(framework::GradVarName("Bias"), InputGrad("Bias"));

    return std::unique_ptr<framework::OpDesc>(op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(batch_norm, ops::BatchNormOp, ops::BatchNormOpMaker,
                  ops::BatchNormGradMaker);
REGISTER_OPERATOR(batch_norm_grad, ops::BatchNormGradOp);

REGISTER_OP_CPU_KERNEL(
    batch_norm, ops::BatchNormKernel<paddle::platform::CPUDeviceContext, float>,
    ops::BatchNormKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    batch_norm_grad,
    ops::BatchNormGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::BatchNormGradKernel<paddle::platform::CPUDeviceContext, double>);
