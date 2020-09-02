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
      ((this->IsMKLDNNType() == true) || (data_layout == DataLayout::kNCHW)
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
      platform::CanMKLDNNBeUsed(ctx)) {
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
      .AsDispensable();
  AddAttr<bool>("use_mkldnn",
                "(bool, default false) Only used in mkldnn kernel")
      .SetDefault(false);
  AddAttr<bool>("fuse_with_relu",
                "(bool, default false) Only used in mkldnn kernel")
      .SetDefault(false);
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

template <typename T>
class BatchNormKernel<platform::CPUDeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const float epsilon = ctx.Attr<float>("epsilon");
    float momentum = ctx.Attr<float>("momentum");
    const bool is_test = ctx.Attr<bool>("is_test");
    const bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    const bool trainable_stats = ctx.Attr<bool>("trainable_statistics");
    bool test_mode = is_test && (!trainable_stats);

    bool global_stats = test_mode || use_global_stats;

    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);

    const auto *x = ctx.Input<Tensor>("X");
    const auto &x_dims = x->dims();
    PADDLE_ENFORCE_GE(
        x_dims.size(), 2,
        platform::errors::InvalidArgument(
            "The size of input X's dimensions should be larger than 1."
            "But received: the size of input X's dimensions is [%d]",
            x_dims.size()));
    PADDLE_ENFORCE_LE(
        x_dims.size(), 5,
        platform::errors::InvalidArgument(
            "The size of input X's dimensions should be less than 6."
            "But received: the size of input X's dimensionss is [%d]",
            x_dims.size()));
    const int N = x_dims[0];
    const int C =
        (data_layout == DataLayout::kNCHW ? x_dims[1]
                                          : x_dims[x_dims.size() - 1]);
    const int sample_size = x->numel() / N / C;

    auto *y = ctx.Output<Tensor>("Y");

    auto *mean_out = ctx.Output<Tensor>("MeanOut");
    auto *variance_out = ctx.Output<Tensor>("VarianceOut");
    auto *saved_mean = ctx.Output<Tensor>("SavedMean");
    auto *saved_variance = ctx.Output<Tensor>("SavedVariance");

    // alloc memory
    y->mutable_data<T>(ctx.GetPlace());
    mean_out->mutable_data<T>(ctx.GetPlace());
    variance_out->mutable_data<T>(ctx.GetPlace());
    saved_mean->mutable_data<T>(ctx.GetPlace());
    saved_variance->mutable_data<T>(ctx.GetPlace());

    if (!global_stats) {
      // saved_xx is use just in this batch of data
      EigenVectorArrayMap<T> saved_mean_e(
          saved_mean->mutable_data<T>(ctx.GetPlace()), C);
      EigenVectorArrayMap<T> saved_variance_e(
          saved_variance->mutable_data<T>(ctx.GetPlace()), C);
      saved_mean_e.setZero();
      saved_variance_e.setZero();

      EigenVectorArrayMap<T> running_mean_arr(
          mean_out->mutable_data<T>(ctx.GetPlace()), C);
      EigenVectorArrayMap<T> running_var_arr(
          variance_out->mutable_data<T>(ctx.GetPlace()), C);

      if ((N * sample_size) == 1) {
        // Only 1 element in normalization dimension,
        // we skip the batch norm calculation, let y = x.
        framework::TensorCopy(*x, ctx.GetPlace(), y);
        return;
      }

      switch (data_layout) {
        case DataLayout::kNCHW: {
          ConstEigenArrayMap<T> x_arr(x->data<T>(), sample_size, N * C);
          for (int nc = 0; nc < N * C; ++nc) {
            saved_mean_e(nc % C) += x_arr.col(nc).sum();
          }
          saved_mean_e /= N * sample_size;
          for (int nc = 0; nc < N * C; ++nc) {
            saved_variance_e(nc % C) +=
                (x_arr.col(nc) - saved_mean_e(nc % C)).matrix().squaredNorm();
          }
          saved_variance_e /= N * sample_size;
          break;
        }
        case DataLayout::kNHWC: {
          ConstEigenArrayMap<T> x_arr(x->data<T>(), C, N * sample_size);
          for (int i = 0; i < N * sample_size; ++i) {
            saved_mean_e += x_arr.col(i);
          }
          saved_mean_e /= N * sample_size;
          for (int i = 0; i < N * sample_size; ++i) {
            saved_variance_e +=
                (x_arr.col(i) - saved_mean_e) * (x_arr.col(i) - saved_mean_e);
          }
          saved_variance_e /= N * sample_size;
          break;
        }
        default:
          PADDLE_THROW("Unknown storage order: %s", data_layout_str);
      }

      // if MomentumTensor is set, use MomentumTensor value, momentum
      // is only used in this training branch
      if (ctx.HasInput("MomentumTensor")) {
        const auto *mom_tensor = ctx.Input<Tensor>("MomentumTensor");
        momentum = mom_tensor->data<float>()[0];
      }

      running_mean_arr =
          running_mean_arr * momentum + saved_mean_e * (1. - momentum);
      running_var_arr =
          running_var_arr * momentum + saved_variance_e * (1. - momentum);
    }

    // use SavedMean and SavedVariance to do normalize
    Eigen::Array<T, Eigen::Dynamic, 1> inv_std(C);
    if (global_stats) {
      ConstEigenVectorArrayMap<T> var_arr(
          ctx.Input<Tensor>("Variance")->data<T>(), C);
      inv_std = (var_arr + epsilon).sqrt().inverse();
    } else {
      EigenVectorArrayMap<T> saved_inv_std(
          ctx.Output<Tensor>("SavedVariance")->data<T>(), C);
      // inverse SavedVariance first, gradient will use it too.
      saved_inv_std = (saved_inv_std + epsilon).inverse().sqrt();
      inv_std = saved_inv_std;
    }
    ConstEigenVectorArrayMap<T> mean_arr(
        global_stats ? ctx.Input<Tensor>("Mean")->data<T>()
                     : ctx.Output<Tensor>("SavedMean")->data<T>(),
        C);

    //   ((x - est_mean) * (inv_var) * scale + bias
    //   formula transform ====>
    //   (x * inv_var * scale) + (bias - est_mean * inv_var * scale)
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");
    ConstEigenVectorArrayMap<T> scale_arr(scale->data<T>(), C);
    ConstEigenVectorArrayMap<T> bias_arr(bias->data<T>(), C);
    Eigen::Array<T, Eigen::Dynamic, 1> new_scale = inv_std * scale_arr;
    Eigen::Array<T, Eigen::Dynamic, 1> new_bias =
        bias_arr - mean_arr * inv_std * scale_arr;

    switch (data_layout) {
      case DataLayout::kNCHW: {
        EigenArrayMap<T> y_arr(y->mutable_data<T>(ctx.GetPlace()), sample_size,
                               N * C);
        ConstEigenArrayMap<T> x_arr(x->data<T>(), sample_size, N * C);
        for (int nc = 0; nc < N * C; ++nc) {
          y_arr.col(nc) = x_arr.col(nc) * new_scale(nc % C) + new_bias(nc % C);
        }
        break;
      }
      case DataLayout::kNHWC: {
        EigenArrayMap<T>(y->mutable_data<T>(ctx.GetPlace()), C,
                         N * sample_size) =
            (ConstEigenArrayMap<T>(x->data<T>(), C, N * sample_size).colwise() *
             new_scale)
                .colwise() +
            new_bias;
        break;
      }
      default:
        PADDLE_THROW("Unknown storage order: %d", data_layout);
    }
  }
};

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
  OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Output",
                 framework::GradVarName("X"), "BatchNormGrad");

  const bool has_scale_grad = ctx->HasOutput(framework::GradVarName("Scale"));
  const bool has_bias_grad = ctx->HasOutput(framework::GradVarName("Bias"));

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
      ((this->IsMKLDNNType() == true) || (data_layout == DataLayout::kNCHW)
           ? x_dims[1]
           : x_dims[x_dims.size() - 1]);

  ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
  // has_scale_grad == has_bias_grad, judge has_scale_grad is enough
  if (has_scale_grad) {
    ctx->SetOutputDim(framework::GradVarName("Scale"), {C});
    ctx->SetOutputDim(framework::GradVarName("Bias"), {C});
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

#ifdef PADDLE_WITH_MKLDNN
  if (library == framework::LibraryType::kPlain &&
      platform::CanMKLDNNBeUsed(ctx)) {
    library = framework::LibraryType::kMKLDNN;
    layout = framework::DataLayout::kMKLDNN;
  }
#endif

  return framework::OpKernelType(
      OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace(), layout,
      library);
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
    const auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");
    const auto *saved_mean = ctx.Input<Tensor>("SavedMean");
    // SavedVariance have been reverted in forward operator
    const auto *saved_inv_variance = ctx.Input<Tensor>("SavedVariance");
    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    const bool is_test = ctx.Attr<bool>("is_test");
    const float epsilon = ctx.Attr<float>("epsilon");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);

    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    // batch_norm with inplace as false will take X as grad input, which
    // is same as cuDNN batch_norm backward calculation, batch_norm
    // with inplace as true only take Y as input and X should be calculate
    // by inverse operation of batch_norm on Y
    const Tensor *x;
    bool is_inplace;
    if (ctx.HasInput("Y")) {
      x = ctx.Input<Tensor>("Y");
      is_inplace = true;
      PADDLE_ENFORCE_EQ(d_x, d_y,
                        platform::errors::InvalidArgument(
                            "X@GRAD and Y@GRAD not inplace in inplace mode"));
    } else {
      x = ctx.Input<Tensor>("X");
      is_inplace = false;
      PADDLE_ENFORCE_NE(d_x, d_y,
                        platform::errors::InvalidArgument(
                            "X@GRAD and Y@GRAD inplaced in non-inplace mode"));
    }

    PADDLE_ENFORCE_EQ(
        is_test, false,
        platform::errors::InvalidArgument(
            "`is_test = True` CANNOT be used in train program. If "
            "you want to use global status in pre_train model, "
            "please set `use_global_stats = True`"));

    // Get the size for each dimension.
    // NCHW [batch_size, in_channels, in_height, in_width]
    const auto &x_dims = x->dims();
    PADDLE_ENFORCE_GE(
        x_dims.size(), 2,
        platform::errors::InvalidArgument(
            "The size of input X's dimensions should be larger than 1."
            "But received: the size of input X's dimensions is [%d]",
            x_dims.size()));
    PADDLE_ENFORCE_LE(
        x_dims.size(), 5,
        platform::errors::InvalidArgument(
            "The size of input X's dimensions should be less than 6."
            "But received: the size of input X's dimensions is [%d]",
            x_dims.size()));
    const int N = x_dims[0];
    const int C =
        (data_layout == DataLayout::kNCHW ? x_dims[1]
                                          : x_dims[x_dims.size() - 1]);
    const int sample_size = x->numel() / N / C;

    // init output
    d_x->mutable_data<T>(ctx.GetPlace());

    const T *mean_data = saved_mean->data<T>();
    const T *inv_var_data = saved_inv_variance->data<T>();
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

    ConstEigenVectorArrayMap<T> scale_arr(scale->data<T>(), C);
    ConstEigenVectorArrayMap<T> bias_arr(bias->data<T>(), C);
    ConstEigenVectorArrayMap<T> mean_arr(mean_data, C);
    ConstEigenVectorArrayMap<T> inv_var_arr(inv_var_data, C);

    T *d_bias_data = nullptr;
    T *d_scale_data = nullptr;
    if (d_scale && d_bias) {
      d_scale->mutable_data<T>(ctx.GetPlace());
      d_bias->mutable_data<T>(ctx.GetPlace());
      d_bias_data = d_bias->mutable_data<T>(ctx.GetPlace());
      d_scale_data = d_scale->mutable_data<T>(ctx.GetPlace());
    }

    // d_bias = np.sum(d_y, axis=0)
    // d_scale = np.sum((X - mean) / inv_std * dy, axis=0)
    // d_x = (1. / N) * scale * inv_var * (N * d_y - np.sum(d_y, axis=0)
    //   - (X - mean) * inv_var * inv_var * np.sum(d_y * (X - mean), axis=0))
    EigenVectorArrayMap<T> d_bias_arr(d_bias_data, C);
    EigenVectorArrayMap<T> d_scale_arr(d_scale_data, C);

    if (d_scale && d_bias) {
      d_bias_arr.setZero();
      d_scale_arr.setZero();
    }

    if ((N * sample_size) == 1 && !use_global_stats) {
      framework::TensorCopy(*d_y, ctx.GetPlace(), d_x);
      return;
    }

    int scale_coefff = use_global_stats ? 1 : N * sample_size;
    const auto scale_inv_var_nhw = scale_arr * inv_var_arr / scale_coefff;

    Tensor dy_sum;
    dy_sum.Resize({C});
    dy_sum.mutable_data<T>(ctx.GetPlace());
    EigenVectorArrayMap<T> dy_sum_arr(dy_sum.mutable_data<T>(ctx.GetPlace()),
                                      C);

    Tensor dy_mul_x_sub_mean_mul_invstd_sum;
    dy_mul_x_sub_mean_mul_invstd_sum.Resize({C});
    dy_mul_x_sub_mean_mul_invstd_sum.mutable_data<T>(ctx.GetPlace());
    EigenVectorArrayMap<T> dy_mul_x_sub_mean_mul_invstd_sum_arr(
        dy_mul_x_sub_mean_mul_invstd_sum.mutable_data<T>(ctx.GetPlace()), C);

    dy_sum_arr.setZero();
    dy_mul_x_sub_mean_mul_invstd_sum_arr.setZero();

    // inplace calculation
    // Y:  ((x - est_mean) * (inv_var) * scale + bias
    //   formula transform ====>
    //   (x * inv_var * scale) + (bias - est_mean * inv_var * scale)
    // X: (y - bias) / scale / (inv_var) + est_mean
    //   formula transform ====>
    //    (y - bias) / (scale * inv_var) + est_mean
    switch (data_layout) {
      case DataLayout::kNCHW: {
        if (is_inplace) {
          auto px = *x;
          EigenArrayMap<T> x_data(px.mutable_data<T>(ctx.GetPlace()),
                                  sample_size, N * C);
          ConstEigenArrayMap<T> y_data(x->data<T>(), sample_size, N * C);
          for (int nc = 0; nc < N * C; ++nc) {
            x_data.col(nc) = (y_data.col(nc) - bias_arr(nc % C)) /
                                 scale_inv_var_nhw(nc % C) / scale_coefff +
                             mean_arr(nc % C);
          }
        }
        ConstEigenArrayMap<T> x_arr(x->data<T>(), sample_size, N * C);
        ConstEigenArrayMap<T> d_y_arr(d_y->data<T>(), sample_size, N * C);
        EigenArrayMap<T> d_x_arr(d_x->mutable_data<T>(ctx.GetPlace()),
                                 sample_size, N * C);

        for (int nc = 0; nc < N * C; ++nc) {
          int c = nc % C;
          dy_sum_arr(c) += d_y_arr.col(nc).sum();
          dy_mul_x_sub_mean_mul_invstd_sum_arr(c) +=
              ((x_arr.col(nc) - mean_arr(c)) * inv_var_arr(c) * d_y_arr.col(nc))
                  .sum();
        }

        if (d_scale && d_bias) {
          d_bias_arr = dy_sum_arr;
          d_scale_arr = dy_mul_x_sub_mean_mul_invstd_sum_arr;
        }

        if (!use_global_stats) {
          for (int nc = 0; nc < N * C; ++nc) {
            int c = nc % C;
            d_x_arr.col(nc) =
                scale_inv_var_nhw(c) *
                (d_y_arr.col(nc) * N * sample_size - dy_sum_arr(c) -
                 (x_arr.col(nc) - mean_arr[c]) *
                     dy_mul_x_sub_mean_mul_invstd_sum_arr(c) * inv_var_arr(c));
          }
        } else {
          for (int nc = 0; nc < N * C; ++nc) {
            int c = nc % C;
            d_x_arr.col(nc) = scale_inv_var_nhw(c) * d_y_arr.col(nc);
          }
        }
        break;
      }
      case DataLayout::kNHWC: {
        if (is_inplace) {
          auto px = *x;
          EigenArrayMap<T> x_data(px.mutable_data<T>(ctx.GetPlace()), C,
                                  N * sample_size);
          ConstEigenArrayMap<T> y_data(x->data<T>(), C, N * sample_size);
          for (int nhw = 0; nhw < N * sample_size; nhw++) {
            x_data.col(nhw) = (y_data.col(nhw) - bias_arr) / scale_inv_var_nhw /
                                  scale_coefff +
                              mean_arr;
          }
        }
        ConstEigenArrayMap<T> x_arr(x->data<T>(), C, N * sample_size);
        ConstEigenArrayMap<T> d_y_arr(d_y->data<T>(), C, N * sample_size);
        EigenArrayMap<T> d_x_arr(d_x->mutable_data<T>(ctx.GetPlace()), C,
                                 N * sample_size);

        for (int nhw = 0; nhw < N * sample_size; ++nhw) {
          dy_sum_arr += d_y_arr.col(nhw);
          dy_mul_x_sub_mean_mul_invstd_sum_arr +=
              (x_arr.col(nhw) - mean_arr) * inv_var_arr * d_y_arr.col(nhw);
        }

        if (d_scale && d_bias) {
          d_bias_arr = dy_sum_arr;
          d_scale_arr = dy_mul_x_sub_mean_mul_invstd_sum_arr;
        }

        if (!use_global_stats) {
          for (int nhw = 0; nhw < N * sample_size; ++nhw) {
            d_x_arr.col(nhw) =
                scale_inv_var_nhw *
                (d_y_arr.col(nhw) * N * sample_size - dy_sum_arr -
                 (x_arr.col(nhw) - mean_arr) *
                     dy_mul_x_sub_mean_mul_invstd_sum_arr * inv_var_arr);
          }
        } else {
          for (int nhw = 0; nhw < N * sample_size; ++nhw) {
            d_x_arr.col(nhw) = scale_inv_var_nhw * d_y_arr.col(nhw);
          }
        }
        break;
      }
      default:
        PADDLE_THROW("Unknown storage order: %s", data_layout_str);
    }
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
  if (BOOST_GET_CONST(bool, this->GetAttr("use_global_stats"))) {
    op->SetInput("Mean", this->Output("MeanOut"));
    op->SetInput("Variance", this->Output("VarianceOut"));
  }

  op->SetAttrMap(this->Attrs());

  op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  op->SetOutput(framework::GradVarName("Scale"), this->InputGrad("Scale"));
  op->SetOutput(framework::GradVarName("Bias"), this->InputGrad("Bias"));
}

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(batch_norm, ops::BatchNormOp, ops::BatchNormOpMaker,
                  ops::BatchNormOpInferVarType,
                  ops::BatchNormGradMaker<paddle::framework::OpDesc>,
                  ops::BatchNormGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(batch_norm_grad, ops::BatchNormGradOp);

REGISTER_OP_CPU_KERNEL(
    batch_norm, ops::BatchNormKernel<paddle::platform::CPUDeviceContext, float>,
    ops::BatchNormKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    batch_norm_grad,
    ops::BatchNormGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::BatchNormGradKernel<paddle::platform::CPUDeviceContext, double>);
