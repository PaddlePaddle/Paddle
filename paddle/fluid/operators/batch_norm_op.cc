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
#include <memory>
#include <string>
#include <unordered_map>
#include "paddle/fluid/framework/data_layout.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

void BatchNormOp::InferShape(framework::InferShapeContext *ctx) const {
  OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "BatchNorm");
  OP_INOUT_CHECK(ctx->HasInput("Scale"), "Input", "Scale", "BatchNorm");
  OP_INOUT_CHECK(ctx->HasInput("Bias"), "Input", "Bias", "BatchNorm");
  OP_INOUT_CHECK(ctx->HasInput("Mean"), "Input", "Mean", "BatchNorm");
  OP_INOUT_CHECK(ctx->HasInput("Variance"), "Input", "Variance", "BatchNorm");
  OP_INOUT_CHECK(ctx->HasOutput("Y"), "Output", "Y", "BatchNorm");

  bool is_test = ctx->Attrs().Get<bool>("is_test");
  bool trainable_stats = ctx->Attrs().Get<bool>("trainable_statistics");
  bool test_mode = is_test && (!trainable_stats);
  if (!test_mode) {
    OP_INOUT_CHECK(ctx->HasOutput("MeanOut"), "Output", "MeanOut", "BatchNorm");
    OP_INOUT_CHECK(ctx->HasOutput("VarianceOut"), "Output", "VarianceOut",
                   "BatchNorm");
    OP_INOUT_CHECK(ctx->HasOutput("SavedMean"), "Output", "SavedMean",
                   "BatchNorm");
    OP_INOUT_CHECK(ctx->HasOutput("SavedVariance"), "Output", "SavedVariance",
                   "BatchNorm");
  }

  // make sure Mean/MeanOut and Variance/VarianceOut share memory in Python
  PADDLE_ENFORCE_EQ(ctx->Inputs("Mean")[0], ctx->Outputs("MeanOut")[0],
                    platform::errors::InvalidArgument(
                        "Mean and MeanOut should share the same memory"));
  PADDLE_ENFORCE_EQ(
      ctx->Inputs("Variance")[0], ctx->Outputs("VarianceOut")[0],
      platform::errors::InvalidArgument(
          "Variance and VarianceOut should share the same memory"));

  const auto x_dims = ctx->GetInputDim("X");

  for (int i = 0; i < x_dims.size(); i++) {
    PADDLE_ENFORCE_EQ(
        (x_dims[i] == -1) || (x_dims[i] > 0), true,
        platform::errors::InvalidArgument(
            "Each dimension of input tensor is expected to be -1 or a "
            "positive number, but recieved %d. Input's shape is [%s].",
            x_dims[i], x_dims));
  }

  const DataLayout data_layout = framework::StringToDataLayout(
      ctx->Attrs().Get<std::string>("data_layout"));

  if (ctx->IsRuntime() && ctx->HasInput("MomentumTensor")) {
    auto mom = ctx->Inputs("MomentumTensor");
    PADDLE_ENFORCE_EQ(mom.size(), 1,
                      platform::errors::InvalidArgument(
                          "The input tensor MomentumTensor's size must be 1"
                          "But received: MomentumTensor's size is [%d]",
                          mom.size()));
  }

  PADDLE_ENFORCE_GE(
      x_dims.size(), 2,
      platform::errors::InvalidArgument(
          "ShapeError: the dimension of input "
          "X must greater than or equal to 2. But received: the shape of input "
          "X = [%s], the dimension of input X =[%d]",
          x_dims, x_dims.size()));
  PADDLE_ENFORCE_LE(
      x_dims.size(), 5,
      platform::errors::InvalidArgument(
          "ShapeError: the dimension of input X "
          "must smaller than or equal to 5. But received: the shape of input X "
          "= [%s], the dimension of input X = [%d]",
          x_dims, x_dims.size()));

  const int64_t C =
      ((ctx->IsRunMKLDNNKernel() == true) || (data_layout == DataLayout::kNCHW)
           ? x_dims[1]
           : x_dims[x_dims.size() - 1]);

  auto scale_dim = ctx->GetInputDim("Scale");
  auto bias_dim = ctx->GetInputDim("Bias");

  PADDLE_ENFORCE_EQ(
      scale_dim.size(), 1UL,
      platform::errors::InvalidArgument(
          "ShapeError: the dimension of scale must equal to 1."
          "But received: the shape of scale is [%s], the dimension "
          "of scale is [%d]",
          scale_dim, scale_dim.size()));
  PADDLE_ENFORCE_EQ(bias_dim.size(), 1UL,
                    platform::errors::InvalidArgument(
                        "ShapeError: the dimension of bias must equal to 1."
                        "But received: the shape of bias is [%s],the dimension "
                        "of bias is [%d]",
                        bias_dim, bias_dim.size()));

  bool check = true;
  if ((!ctx->IsRuntime()) && (framework::product(scale_dim) <= 0 ||
                              framework::product(bias_dim) <= 0)) {
    check = false;
  }

  if (check) {
    PADDLE_ENFORCE_EQ(scale_dim[0], C,
                      platform::errors::InvalidArgument(
                          "ShapeError: the shape of scale must equal to [%d]"
                          "But received: the shape of scale is [%d]",
                          C, scale_dim[0]));
    PADDLE_ENFORCE_EQ(bias_dim[0], C,
                      platform::errors::InvalidArgument(
                          "ShapeError: the shape of bias must equal to [%d]"
                          "But received: the shape of bias is [%d]",
                          C, bias_dim[0]));
  }
  ctx->SetOutputDim("Y", x_dims);
  ctx->SetOutputDim("MeanOut", {C});
  ctx->SetOutputDim("VarianceOut", {C});
  ctx->SetOutputDim("SavedMean", {C});
  ctx->SetOutputDim("SavedVariance", {C});
  ctx->ShareLoD("X", "Y");
}

framework::OpKernelType BatchNormOp::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
  auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
  // By default, the type of the scale, bias, mean,
  // and var tensors should both be float. (For float or float16 input tensor)
  // or double (For double input tensor).
  auto bn_param_type = framework::proto::VarType::FP32;
  if (input_data_type == framework::proto::VarType::FP64) {
    bn_param_type = framework::proto::VarType::FP64;
  }
  PADDLE_ENFORCE_EQ(
      bn_param_type, ctx.Input<Tensor>("Scale")->type(),
      platform::errors::InvalidArgument("Scale input should be of float type"));
  PADDLE_ENFORCE_EQ(
      bn_param_type, ctx.Input<Tensor>("Bias")->type(),
      platform::errors::InvalidArgument("Bias input should be of float type"));
  PADDLE_ENFORCE_EQ(
      bn_param_type, ctx.Input<Tensor>("Mean")->type(),
      platform::errors::InvalidArgument("Mean input should be of float type"));
  PADDLE_ENFORCE_EQ(bn_param_type, ctx.Input<Tensor>("Variance")->type(),
                    platform::errors::InvalidArgument(
                        "Variance input should be of float type"));

  // TODO(pzelazko-intel): enable MKLDNN layout when it's ready
  framework::LibraryType library = framework::LibraryType::kPlain;
  framework::DataLayout layout = framework::DataLayout::kAnyLayout;
#ifdef PADDLE_WITH_MKLDNN
  if (library == framework::LibraryType::kPlain &&
      this->CanMKLDNNBeUsed(ctx, input_data_type)) {
    library = framework::LibraryType::kMKLDNN;
    layout = framework::DataLayout::kMKLDNN;
  }
#endif

  return framework::OpKernelType(input_data_type, ctx.GetPlace(), layout,
                                 library);
}

framework::OpKernelType BatchNormOp::GetKernelTypeForVar(
    const std::string &var_name, const Tensor &tensor,
    const framework::OpKernelType &expected_kernel_type) const {
#ifdef PADDLE_WITH_MKLDNN
  // Only input require reshaping, weights and
  // bias are having shape in NCHW order
  if ((var_name == "X") &&
      (expected_kernel_type.data_layout_ == framework::DataLayout::kMKLDNN) &&
      (tensor.layout() != framework::DataLayout::kMKLDNN)) {
    auto attrs = Attrs();
    auto ar = paddle::framework::AttrReader(attrs);
    const std::string data_layout = ar.Get<std::string>("data_layout");
    auto dl = framework::StringToDataLayout(data_layout);
    // Some models may have intentionally set "AnyLayout" for pool
    // op. Treat this as NCHW (default data_format value)
    if (dl != framework::DataLayout::kAnyLayout) {
      return framework::OpKernelType(expected_kernel_type.data_type_,
                                     tensor.place(), dl);
    }
  }
#endif
  return framework::OpKernelType(expected_kernel_type.data_type_,
                                 tensor.place(), tensor.layout());
}

void BatchNormOpMaker::Make() {
  AddAttr<bool>("is_test",
                "(bool, default false) Set to true for inference only, false "
                "for training. Some layers may run faster when this is true.")
      .SetDefault(false);
  AddAttr<float>("momentum", "").SetDefault(0.9);
  AddAttr<float>("epsilon", "")
      .SetDefault(1e-5)
      .AddCustomChecker([](const float &epsilon) {
        PADDLE_ENFORCE_GE(
            epsilon, 0.0f,
            platform::errors::InvalidArgument(
                "'epsilon' should be greater or equal than 0.0."));
        PADDLE_ENFORCE_LE(epsilon, 0.001f,
                          platform::errors::InvalidArgument(
                              "'epsilon' should be less or equal than 0.001."));
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
  AddInput("MomentumTensor",
           "(Tensor<float32>, optional) If provided, batch_norm will "
           "use this as momentum, this has a higher priority than "
           "attr(momentum), the shape of this tensor MUST BE [1].")
      .AsDispensable();
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
  AddOutput("ReserveSpace",
            "Reserve GPU space for triggering the new semi-persistent "
            "NHWC kernel")
      .AsDispensable()
      .AsExtra();
  AddAttr<bool>("use_mkldnn",
                "(bool, default false) Only used in mkldnn kernel")
      .SetDefault(false)
      .AsExtra();
  AddAttr<bool>("fuse_with_relu",
                "(bool, default false) Only used in mkldnn kernel")
      .SetDefault(false)
      .AsExtra();
  AddAttr<bool>("use_global_stats",
                "(bool, default false) Whether to use global mean and "
                "variance. In inference or test mode, set use_global_stats "
                "to true or is_test true. the behavior is equivalent. "
                "In train mode, when setting use_global_stats True, the "
                "global mean and variance are also used during train time, "
                "the BN acts as scaling and shiffting.")
      .SetDefault(false);
  AddAttr<bool>("trainable_statistics",
                "(bool, default false) Whether to calculate mean and variance "
                "in test mode. If setting true in test mode, mean and variace "
                "will be calculated by current batch statistics.")
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


void BatchNormGradOp::InferShape(framework::InferShapeContext *ctx) const {
  // check input
  OP_INOUT_CHECK(ctx->HasInput("Scale"), "Input", "Scale", "BatchNormGrad");
  OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Y")), "Input",
                 framework::GradVarName("Y"), "BatchNormGrad");
  OP_INOUT_CHECK(ctx->HasInput("SavedMean"), "Input", "SavedMean",
                 "BatchNormGrad");
  OP_INOUT_CHECK(ctx->HasInput("SavedVariance"), "Input", "SavedVariance",
                 "BatchNormGrad");

  // check output
  const bool has_scale_grad = ctx->HasOutput(framework::GradVarName("Scale"));
  const bool has_bias_grad = ctx->HasOutput(framework::GradVarName("Bias"));
  const bool has_x_grad = ctx->HasOutput(framework::GradVarName("X"));

  PADDLE_ENFORCE_EQ((has_scale_grad == has_bias_grad), true,
                    platform::errors::NotFound(
                        "Output(Scale@GRAD) and Output(Bias@GRAD) must be null "
                        "or not be null at same time. But now, "
                        "has Scale@Grad=[%d], has Bias@GRAD=[%d]",
                        has_scale_grad, has_bias_grad));

  const bool use_global_stats = ctx->Attrs().Get<bool>("use_global_stats");
  if (use_global_stats) {
    PADDLE_ENFORCE_EQ(
        !ctx->Attrs().Get<bool>("use_mkldnn"), true,
        platform::errors::InvalidArgument(
            "Using global stats during training is not supported "
            "in gradient op kernel of batch_norm_mkldnn_op now."));
  }

  OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "BatchNormGrad");
  const auto x_dims = ctx->GetInputDim("X");
  const DataLayout data_layout = framework::StringToDataLayout(
      ctx->Attrs().Get<std::string>("data_layout"));

  const int C =
      ((ctx->IsRunMKLDNNKernel() == true) || (data_layout == DataLayout::kNCHW)
           ? x_dims[1]
           : x_dims[x_dims.size() - 1]);

  // has_scale_grad == has_bias_grad, judge has_scale_grad is enough
  if (has_scale_grad) {
    ctx->SetOutputDim(framework::GradVarName("Scale"), {C});
    ctx->SetOutputDim(framework::GradVarName("Bias"), {C});
  }
  if (has_x_grad) {
    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
  }
}

framework::OpKernelType BatchNormGradOp::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
  const auto *var = ctx.InputVar(framework::GradVarName("Y"));
  if (var == nullptr) {
    PADDLE_THROW(
        platform::errors::InvalidArgument("can't find gradient variable of Y"));
  }
  const Tensor *t = nullptr;
  if (var->IsType<Tensor>()) {
    t = &var->Get<Tensor>();
  } else if (var->IsType<LoDTensor>()) {
    t = &var->Get<LoDTensor>();
  }
  if (t == nullptr) {
    PADDLE_THROW(
        platform::errors::InvalidArgument("gradient variable of Y is empty"));
  }

  // TODO(pzelazko-intel): enable MKLDNN layout when it's ready
  framework::LibraryType library = framework::LibraryType::kPlain;
  framework::DataLayout layout = framework::DataLayout::kAnyLayout;
  auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");

#ifdef PADDLE_WITH_MKLDNN
  if (library == framework::LibraryType::kPlain &&
      this->CanMKLDNNBeUsed(ctx, data_type)) {
    library = framework::LibraryType::kMKLDNN;
    layout = framework::DataLayout::kMKLDNN;
  }
#endif

  return framework::OpKernelType(data_type, ctx.GetPlace(), layout, library);
}

framework::OpKernelType BatchNormGradOp::GetKernelTypeForVar(
    const std::string &var_name, const Tensor &tensor,
    const framework::OpKernelType &expected_kernel_type) const {
#ifdef PADDLE_WITH_MKLDNN
  // Only input require reshaping, weights and
  // bias are having shape in NCHW order
  if (((var_name == "X") || (var_name == framework::GradVarName("Y"))) &&
      (expected_kernel_type.data_layout_ == framework::DataLayout::kMKLDNN) &&
      (tensor.layout() != framework::DataLayout::kMKLDNN)) {
    auto attrs = Attrs();
    auto ar = paddle::framework::AttrReader(attrs);
    const std::string data_layout = ar.Get<std::string>("data_layout");
    auto dl = framework::StringToDataLayout(data_layout);
    // Some models may have intentionally set "AnyLayout" for pool
    // op. Treat this as NCHW (default data_format value)
    if (dl != framework::DataLayout::kAnyLayout) {
      return framework::OpKernelType(expected_kernel_type.data_type_,
                                     tensor.place(), dl);
    }
  }
#endif
  return framework::OpKernelType(expected_kernel_type.data_type_,
                                 tensor.place(), tensor.layout());
}

template <typename T>
class BatchNormGradKernel<platform::CPUDeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {

  }
};

template <typename T>
void BatchNormGradMaker<T>::Apply(GradOpPtr<T> op) const {
  op->SetType(this->ForwardOpType() + "_grad");
  op->SetInput("X", this->Input("X"));
  op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));

  op->SetInput("Scale", this->Input("Scale"));
  op->SetInput("Bias", this->Input("Bias"));
  op->SetInput("SavedMean", this->Output("SavedMean"));
  op->SetInput("SavedVariance", this->Output("SavedVariance"));
  if (this->HasOutput("ReserveSpace")) {
    op->SetInput("ReserveSpace", this->Output("ReserveSpace"));
  }

  // used when setting use_global_stats True during training
  if (BOOST_GET_CONST(bool, this->GetAttr("use_global_stats")) ||
      BOOST_GET_CONST(bool, this->GetAttr("is_test"))) {
    op->SetInput("Mean", this->Output("MeanOut"));
    op->SetInput("Variance", this->Output("VarianceOut"));
  }

  op->SetAttrMap(this->Attrs());

  op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  op->SetOutput(framework::GradVarName("Scale"), this->InputGrad("Scale"));
  op->SetOutput(framework::GradVarName("Bias"), this->InputGrad("Bias"));
}

template <typename T>
void BatchNormDoubleGradMaker<T>::Apply(GradOpPtr<T> op) const {
  op->SetType("batch_norm_grad_grad");
  op->SetInput("X", this->Input("X"));
  op->SetInput("Scale", this->Input("Scale"));
  op->SetInput("SavedMean", this->Input("SavedMean"));
  op->SetInput("SavedVariance", this->Input("SavedVariance"));
  if (BOOST_GET_CONST(bool, this->GetAttr("use_global_stats"))) {
    op->SetInput("Mean", this->Input("Mean"));
    op->SetInput("Variance", this->Input("Variance"));
  }
  op->SetInput("DDX", this->OutputGrad(framework::GradVarName("X")));
  op->SetInput("DDScale", this->OutputGrad(framework::GradVarName("Scale")));
  op->SetInput("DDBias", this->OutputGrad(framework::GradVarName("Bias")));
  op->SetInput("DY", this->Input(framework::GradVarName("Y")));

  op->SetAttrMap(this->Attrs());
  op->SetOutput("DX", this->InputGrad("X"));
  op->SetOutput("DScale", this->InputGrad("Scale"));
  op->SetOutput("DDY", this->InputGrad(framework::GradVarName("Y")));
}

void BatchNormDoubleGradOp::InferShape(
    framework::InferShapeContext *ctx) const {
  OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "BatchNormDoubleGrad");
  OP_INOUT_CHECK(ctx->HasInput("Scale"), "Input", "Scale",
                 "BatchNormDoubleGrad");
  OP_INOUT_CHECK(ctx->HasInput("SavedMean"), "Input", "SavedMean",
                 "BatchNormDoubleGrad");
  OP_INOUT_CHECK(ctx->HasInput("SavedVariance"), "Input", "SavedVariance",
                 "BatchNormDoubleGrad");

  const bool use_global_stats = ctx->Attrs().Get<bool>("use_global_stats");
  if (use_global_stats) {
    OP_INOUT_CHECK(ctx->HasInput("Variance"), "Input", "VarianceOut",
                   "BatchNormDoubleGrad");
  }

  OP_INOUT_CHECK(ctx->HasInput("DY"), "Input", "DY", "BatchNormDoubleGrad");

  // check output
  OP_INOUT_CHECK(ctx->HasOutput("DX"), "Output", "DX", "BatchNormDoubleGrad");

  const auto x_dims = ctx->GetInputDim("X");
  const DataLayout data_layout = framework::StringToDataLayout(
      ctx->Attrs().Get<std::string>("data_layout"));
  const int C =
      ((ctx->IsRunMKLDNNKernel() == true) || (data_layout == DataLayout::kNCHW)
           ? x_dims[1]
           : x_dims[x_dims.size() - 1]);

  if (ctx->HasOutput("DX")) {
    ctx->SetOutputDim("DX", x_dims);
  }
  if (ctx->HasOutput("DScale")) {
    ctx->SetOutputDim("DScale", {C});
  }
  if (ctx->HasOutput("DDY")) {
    ctx->ShareDim("X", "DDY");
  }
}

framework::OpKernelType BatchNormDoubleGradOp::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
  const auto *var = ctx.InputVar("DY");
  if (var == nullptr) {
    PADDLE_THROW(
        platform::errors::NotFound("cannot find gradient variable of Y"));
  }
  const Tensor *t = nullptr;
  if (var->IsType<Tensor>()) {
    t = &var->Get<Tensor>();
  } else if (var->IsType<LoDTensor>()) {
    t = &var->Get<LoDTensor>();
  }
  if (t == nullptr) {
    PADDLE_THROW(
        platform::errors::InvalidArgument("gradient variable of Y is empty"));
  }
  return framework::OpKernelType(
      OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
}

template <typename T>
class BatchNormDoubleGradKernel<platform::CPUDeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *X = ctx.Input<Tensor>("X");
    const auto *Scale = ctx.Input<Tensor>("Scale");
    const auto *dY = ctx.Input<Tensor>("DY");
    const auto *Saved_mean = ctx.Input<Tensor>("SavedMean");
    const auto *Saved_variance = ctx.Input<Tensor>("SavedVariance");
    const float epsilon = ctx.Attr<float>("epsilon");
    const bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    const bool is_test = ctx.Attr<bool>("is_test");

    PADDLE_ENFORCE_EQ(
        is_test, false,
        platform::errors::InvalidArgument(
            "`is_test = True` CANNOT be used in train program. If "
            "you want to use global status in pre_train model, "
            "please set `use_global_stats = True`"));

    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);

    const auto *ddX = ctx.Input<Tensor>("DDX");
    const auto *ddScale = ctx.Input<Tensor>("DDScale");
    const auto *ddBias = ctx.Input<Tensor>("DDBias");

    auto *dX = ctx.Output<Tensor>("DX");
    auto *dScale = ctx.Output<Tensor>("DScale");
    auto *ddY = ctx.Output<Tensor>("DDY");
    dX->mutable_data<T>(ctx.GetPlace());
    ddY->mutable_data<T>(ctx.GetPlace());

    auto &dev_ctx = ctx.template device_context<platform::CPUDeviceContext>();

    const auto &x_dims = X->dims();
    const int C =
        (data_layout == DataLayout::kNCHW ? x_dims[1]
                                          : x_dims[x_dims.size() - 1]);
    const int sample_size = X->numel() / C;
    pten::funcs::SetConstant<platform::CPUDeviceContext, T> set_constant;

    const T *mean_data = Saved_mean->data<T>();
    const T *inv_var_data = Saved_variance->data<T>();

    Tensor inv_var_tensor;
    if (use_global_stats) {
      const auto *running_mean = ctx.Input<Tensor>("Mean");
      const auto *running_variance = ctx.Input<Tensor>("Variance");
      mean_data = running_mean->data<T>();
      inv_var_tensor.Resize({C});

      T *running_inv_var_data = inv_var_tensor.mutable_data<T>(ctx.GetPlace());
      EigenVectorArrayMap<T> inv_var_tmp(running_inv_var_data, C);
      ConstEigenVectorArrayMap<T> var_arr(running_variance->data<T>(), C);

      inv_var_tmp = (var_arr + epsilon).sqrt().inverse();
      inv_var_data = running_inv_var_data;
    }

    // transpose NCHW -> NHWC for easy calculate
    Tensor transformed_x(X->type());
    Tensor transformed_dy(dY->type());
    Tensor transformed_ddx(ddX->type());

    Tensor transformed_dx(dX->type());
    Tensor transformed_ddy(ddY->type());
    if (data_layout == DataLayout::kNCHW && x_dims.size() > 2) {
      VLOG(3) << "Transform batchnorm output from NCHW to NHWC";
      // Input Tensor
      ResizeToChannelLast<platform::CPUDeviceContext, T>(ctx, X,
                                                         &transformed_x);
      TransToChannelLast<platform::CPUDeviceContext, T>(ctx, X, &transformed_x);
      ResizeToChannelLast<platform::CPUDeviceContext, T>(ctx, dY,
                                                         &transformed_dy);
      TransToChannelLast<platform::CPUDeviceContext, T>(ctx, dY,
                                                        &transformed_dy);
      ResizeToChannelLast<platform::CPUDeviceContext, T>(ctx, ddX,
                                                         &transformed_ddx);
      TransToChannelLast<platform::CPUDeviceContext, T>(ctx, ddX,
                                                        &transformed_ddx);
      // Output Tensor
      ResizeToChannelLast<platform::CPUDeviceContext, T>(ctx, dX,
                                                         &transformed_dx);
      ResizeToChannelLast<platform::CPUDeviceContext, T>(ctx, ddY,
                                                         &transformed_ddy);
    } else {
      transformed_x.ShareDataWith(*X);
      transformed_dy.ShareDataWith(*dY);
      transformed_ddx.ShareDataWith(*ddX);

      transformed_dx.ShareDataWith(*dX);
      transformed_ddy.ShareDataWith(*ddY);
    }

    ConstEigenArrayMap<T> x_arr(transformed_x.data<T>(), C, sample_size);
    ConstEigenVectorArrayMap<T> mean_arr(mean_data, C);
    ConstEigenVectorArrayMap<T> inv_var_arr(inv_var_data, C);

    Tensor mean_tile;
    mean_tile.Resize({C, sample_size});
    mean_tile.mutable_data<T>(ctx.GetPlace());
    EigenArrayMap<T> mean_tile_data(mean_tile.mutable_data<T>(ctx.GetPlace()),
                                    C, sample_size);

    Tensor inv_var_tile;
    inv_var_tile.Resize({C, sample_size});
    inv_var_tile.mutable_data<T>(ctx.GetPlace());
    EigenArrayMap<T> inv_var_tile_data(
        inv_var_tile.mutable_data<T>(ctx.GetPlace()), C, sample_size);

    mean_tile_data = mean_arr.replicate(1, sample_size);
    inv_var_tile_data = inv_var_arr.replicate(1, sample_size);

    Tensor Scale_data;
    if (!Scale) {
      Scale_data.mutable_data<T>({C}, ctx.GetPlace());
      set_constant(dev_ctx, &Scale_data, static_cast<T>(1));
    }
    ConstEigenVectorArrayMap<T> scale_arr(
        Scale ? Scale->data<T>() : Scale_data.data<T>(), C);

    Tensor scale_tile;
    scale_tile.Resize({C, sample_size});
    scale_tile.mutable_data<T>(ctx.GetPlace());
    EigenArrayMap<T> scale_tile_data(scale_tile.mutable_data<T>(ctx.GetPlace()),
                                     C, sample_size);
    scale_tile_data = scale_arr.replicate(1, sample_size);

    ConstEigenArrayMap<T> dy_arr(transformed_dy.data<T>(), C, sample_size);
    ConstEigenArrayMap<T> ddx_arr(transformed_ddx.data<T>(), C, sample_size);

    Tensor x_sub_mean_mul_invstd;
    x_sub_mean_mul_invstd.Resize({C, sample_size});
    x_sub_mean_mul_invstd.mutable_data<T>(ctx.GetPlace());
    EigenArrayMap<T> x_sub_mean_mul_invstd_arr(
        x_sub_mean_mul_invstd.mutable_data<T>(ctx.GetPlace()), C, sample_size);
    x_sub_mean_mul_invstd_arr = (x_arr - mean_tile_data) * inv_var_tile_data;

    if (dX) {
      dX->mutable_data<T>(ctx.GetPlace());
      EigenArrayMap<T> dx_arr(transformed_dx.mutable_data<T>(ctx.GetPlace()), C,
                              sample_size);
      dx_arr.setZero();
      if (use_global_stats) {
        // math: dx = (ddscale * dy) * inv_var
        if (ddScale) {
          ConstEigenVectorArrayMap<T> ddscale_arr(ddScale->data<T>(), C);
          Tensor ddscale_tile;
          ddscale_tile.Resize({C, sample_size});
          EigenArrayMap<T> ddscale_tile_data(
              ddscale_tile.mutable_data<T>(ctx.GetPlace()), C, sample_size);
          ddscale_tile_data = ddscale_arr.replicate(1, sample_size);

          dx_arr = dy_arr * ddscale_tile_data * inv_var_tile_data;
        }
      } else {
        // math: dx = scale * ((x - mean) * inv_var / NxHxW * (np.mean(ddx,
        // axis=(n,h,w)) *
        //          np.sum(dy, axis=(n,h,w)) -
        //          np.sum(dy * ddx, axis=(n,h,w)) + 3 * np.mean(dy * (x -
        //          mean),
        //          axis=(n,h,w)) * inv_var.pow(2) *
        //          np.sum(ddx * (x - mean), axis=(n,h,w))) + inv_var.pow(3) /
        //          NxHxW *
        //          np.sum(ddx * (x - mean)) *
        //          (np.mean(dy, axis=(n,h,w)) - dy) + inv_var.pow(3) / NxHxW *
        //          np.sum(dy,
        //          axis=(n,h,w)) * (x - mean) *
        //          (np.mean(ddx, axis=(n,h,w)) - ddx)) + ddr * (dy * inv_var -
        //          inv_var
        //          *
        //          np.mean(dy, axis=(n,h,w)) -
        //          inv_var.pow(3) * (x - mean) * np.mean(dy * (x - mean),
        //          axis=(n,h,w)))

        if (ddX) {
          dx_arr +=
              (x_sub_mean_mul_invstd_arr * inv_var_tile_data *
               inv_var_tile_data / sample_size)
                  .colwise() *
              (ddx_arr.rowwise().sum() * dy_arr.rowwise().sum() / sample_size -
               (dy_arr * ddx_arr).rowwise().sum() +
               3. * (dy_arr * x_sub_mean_mul_invstd_arr).rowwise().sum() *
                   (ddx_arr * x_sub_mean_mul_invstd_arr).rowwise().sum() /
                   sample_size);

          dx_arr += (inv_var_tile_data * inv_var_tile_data).colwise() *
                    (ddx_arr * x_sub_mean_mul_invstd_arr).rowwise().sum() /
                    sample_size *
                    (dy_arr.rowwise().sum() / sample_size - dy_arr);

          dx_arr += (inv_var_tile_data * inv_var_tile_data).colwise() *
                    (dy_arr * x_sub_mean_mul_invstd_arr).rowwise().sum() /
                    sample_size *
                    (ddx_arr.rowwise().sum() / sample_size - ddx_arr);

          dx_arr = scale_tile_data * dx_arr;
        }
        if (ddScale) {
          ConstEigenVectorArrayMap<T> ddscale_arr(ddScale->data<T>(), C);
          Tensor ddscale_tile;
          ddscale_tile.Resize({C, sample_size});
          EigenArrayMap<T> ddscale_tile_data(
              ddscale_tile.mutable_data<T>(ctx.GetPlace()), C, sample_size);
          ddscale_tile_data = ddscale_arr.replicate(1, sample_size);

          dx_arr += (dy_arr * inv_var_tile_data -
                     (dy_arr.rowwise().sum().replicate(1, sample_size) /
                      sample_size) *
                         inv_var_tile_data -
                     x_sub_mean_mul_invstd_arr * inv_var_tile_data *
                         (dy_arr * x_sub_mean_mul_invstd_arr)
                             .rowwise()
                             .sum()
                             .replicate(1, sample_size) /
                         sample_size) *
                    ddscale_tile_data;
        }
      }
      if (data_layout == DataLayout::kNCHW) {
        VLOG(3) << "Transform batchnorm output from NHWC to NCHW";
        TransToChannelFirst<paddle::platform::CPUDeviceContext, T>(
            ctx, &transformed_dx, dX);
      }
    }
    if (dScale) {
      dScale->mutable_data<T>(ctx.GetPlace());
      EigenVectorArrayMap<T> dscale_arr(dScale->mutable_data<T>(ctx.GetPlace()),
                                        C);
      dscale_arr.setZero();
      if (use_global_stats) {
        // math: dscale = np.sum(ddx * dy, axis=(n,h,w)) * inv_var
        if (ddX) {
          dscale_arr = (ddx_arr * dy_arr * inv_var_tile_data).rowwise().sum();
        }
      } else {
        // math: dscale = inv_var * (dy - np.mean(dy, axis=(n,h,w) - (x-mean) *
        //            inv_var.pow(2) * np.mean(dy * (x-mean), axis=(n,h,w)))) *
        //            ddx
        if (ddX) {
          Tensor first_grad;
          first_grad.Resize({C, sample_size});
          EigenArrayMap<T> first_grad_arr(
              first_grad.mutable_data<T>(ctx.GetPlace()), C, sample_size);
          first_grad_arr.setZero();

          first_grad_arr +=
              inv_var_tile_data *
              (dy_arr -
               dy_arr.rowwise().sum().replicate(1, sample_size) / sample_size -
               x_sub_mean_mul_invstd_arr *
                   (dy_arr * x_sub_mean_mul_invstd_arr)
                       .rowwise()
                       .sum()
                       .replicate(1, sample_size) /
                   sample_size);
          dscale_arr = (first_grad_arr * ddx_arr).rowwise().sum();
        }
      }
    }

    if (ddY) {
      ddY->mutable_data<T>(ctx.GetPlace());
      EigenArrayMap<T> ddy_arr(transformed_ddy.mutable_data<T>(ctx.GetPlace()),
                               C, sample_size);
      ddy_arr.setZero();
      if (use_global_stats) {
        // math: ddy = r * ddx * inv_var + ddbias +
        //           ddscale * (x - mean) * inv_var
        if (ddX) {
          ddy_arr = scale_tile_data * ddx_arr * inv_var_tile_data;
        }
      } else {
        // math: ddy = (x - mean) * inv_var * ddscale + ddbias +
        //           scale * inv_var * (ddx - (x - mean) * inv_var.pow(2) *
        //           np.mean(ddx * (x - mean), axis=(n,h,w)))
        if (ddX) {
          ddy_arr +=
              scale_tile_data * inv_var_tile_data *
              (ddx_arr -
               ddx_arr.rowwise().sum().replicate(1, sample_size) / sample_size -
               x_sub_mean_mul_invstd_arr *
                   (ddx_arr * x_sub_mean_mul_invstd_arr)
                       .rowwise()
                       .sum()
                       .replicate(1, sample_size) /
                   sample_size);
        }
      }
      if (ddScale) {
        ConstEigenVectorArrayMap<T> ddscale_arr(ddScale->data<T>(), C);
        Tensor ddscale_tile;
        ddscale_tile.Resize({C, sample_size});
        EigenArrayMap<T> ddscale_tile_data(
            ddscale_tile.mutable_data<T>(ctx.GetPlace()), C, sample_size);
        ddscale_tile_data = ddscale_arr.replicate(1, sample_size);

        ddy_arr += x_sub_mean_mul_invstd_arr * ddscale_tile_data;
      }

      if (ddBias) {
        ConstEigenVectorArrayMap<T> ddbias_arr(ddBias->data<T>(), C);
        Tensor ddbias_tile;
        ddbias_tile.Resize({C, sample_size});
        EigenArrayMap<T> ddbias_tile_data(
            ddbias_tile.mutable_data<T>(ctx.GetPlace()), C, sample_size);
        ddbias_tile_data = ddbias_arr.replicate(1, sample_size);

        ddy_arr += ddbias_tile_data;
      }

      if (data_layout == DataLayout::kNCHW) {
        VLOG(3) << "Transform batchnorm output from NHWC to NCHW";
        TransToChannelFirst<paddle::platform::CPUDeviceContext, T>(
            ctx, &transformed_ddy, ddY);
      }
    }
  }
};

DECLARE_INPLACE_OP_INFERER(BatchNormDoubleGradOpInplaceInferer, {"DY", "DDY"});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(batch_norm, ops::BatchNormOp, ops::BatchNormOpMaker,
                  ops::BatchNormOpInferVarType,
                  ops::BatchNormGradMaker<paddle::framework::OpDesc>,
                  ops::BatchNormGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(batch_norm_grad, ops::BatchNormGradOp,
                  ops::BatchNormDoubleGradMaker<paddle::framework::OpDesc>,
                  ops::BatchNormDoubleGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(batch_norm_grad_grad, ops::BatchNormDoubleGradOp,
                  ops::BatchNormDoubleGradOpInplaceInferer);

// REGISTER_OP_CPU_KERNEL(
//     batch_norm_grad,
//     ops::BatchNormGradKernel<paddle::platform::CPUDeviceContext, float>,
//     ops::BatchNormGradKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    batch_norm_grad_grad,
    ops::BatchNormDoubleGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::BatchNormDoubleGradKernel<paddle::platform::CPUDeviceContext, double>);
