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
  PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) of ConvOp should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("Scale"),
                 "Input(Scale) of ConvOp should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("Bias"),
                 "Input(Bias) of ConvOp should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("Mean"),
                 "Input(Mean) of ConvOp should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("Variance"),
                 "Input(Variance) of ConvOp should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("Y"),
                 "Output(Y) of ConvOp should not be null.");
  bool is_test = ctx->Attrs().Get<bool>("is_test");
  if (!is_test) {
    PADDLE_ENFORCE(ctx->HasOutput("MeanOut"),
                   "Output(MeanOut) of ConvOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("VarianceOut"),
                   "Output(VarianceOut) of ConvOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("SavedMean"),
                   "Output(SavedMean) of ConvOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("SavedVariance"),
                   "Output(SavedVariance) of ConvOp should not be null.");
  }

  // make sure Mean/MeanOut and Variance/VarianceOut share memory in Python
  PADDLE_ENFORCE_EQ(ctx->Inputs("Mean")[0], ctx->Outputs("MeanOut")[0],
                    "Mean and MeanOut should share the same memory");
  PADDLE_ENFORCE_EQ(ctx->Inputs("Variance")[0], ctx->Outputs("VarianceOut")[0],
                    "Variance and VarianceOut should share the same memory");

  const auto x_dims = ctx->GetInputDim("X");
  const DataLayout data_layout = framework::StringToDataLayout(
      ctx->Attrs().Get<std::string>("data_layout"));

  PADDLE_ENFORCE(x_dims.size() >= 2 && x_dims.size() <= 5,
                 "Input X must have 2 to 5 dimensions.");

  const int64_t C =
      (data_layout == DataLayout::kNCHW ? x_dims[1]
                                        : x_dims[x_dims.size() - 1]);

  PADDLE_ENFORCE_EQ(ctx->GetInputDim("Scale").size(), 1UL);
  PADDLE_ENFORCE_EQ(ctx->GetInputDim("Scale")[0], C);
  PADDLE_ENFORCE_EQ(ctx->GetInputDim("Bias").size(), 1UL);
  PADDLE_ENFORCE_EQ(ctx->GetInputDim("Bias")[0], C);

  ctx->SetOutputDim("Y", x_dims);
  ctx->SetOutputDim("MeanOut", {C});
  ctx->SetOutputDim("VarianceOut", {C});
  ctx->SetOutputDim("SavedMean", {C});
  ctx->SetOutputDim("SavedVariance", {C});
  ctx->ShareLoD("X", "Y");
}

framework::OpKernelType BatchNormOp::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
  auto input_data_type = ctx.Input<Tensor>("X")->type();
  // By default, the type of the scale, bias, mean,
  // and var tensors should both be float. (For float or float16 input tensor)
  // or double (For double input tensor).
  auto bn_param_type = framework::proto::VarType::FP32;
  if (input_data_type == framework::proto::VarType::FP64) {
    bn_param_type = framework::proto::VarType::FP64;
  }
  PADDLE_ENFORCE_EQ(bn_param_type, ctx.Input<Tensor>("Scale")->type(),
                    "Scale input should be of float type");
  PADDLE_ENFORCE_EQ(bn_param_type, ctx.Input<Tensor>("Bias")->type(),
                    "Bias input should be of float type");
  PADDLE_ENFORCE_EQ(bn_param_type, ctx.Input<Tensor>("Mean")->type(),
                    "Mean input should be of float type");
  PADDLE_ENFORCE_EQ(bn_param_type, ctx.Input<Tensor>("Variance")->type(),
                    "Variance input should be of float type");

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

void BatchNormOpMaker::Make() {
  AddAttr<bool>("is_test",
                "(bool, default false) Set to true for inference only, false "
                "for training. Some layers may run faster when this is true.")
      .SetDefault(false);
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
    const float momentum = ctx.Attr<float>("momentum");
    const bool is_test = ctx.Attr<bool>("is_test");
    const bool use_global_stats = ctx.Attr<bool>("use_global_stats");

    bool global_stats = is_test || use_global_stats;

    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);

    const auto *x = ctx.Input<Tensor>("X");
    const auto &x_dims = x->dims();
    PADDLE_ENFORCE(x_dims.size() >= 2 && x_dims.size() <= 5,
                   "The Input dim size should be between 2 and 5");
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
        LOG(WARNING) << "Only 1 element in normalization dimension, "
                     << "we skip the batch norm calculation, let y = x.";
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
  PADDLE_ENFORCE(ctx->HasInput("X"));
  PADDLE_ENFORCE(ctx->HasInput("Scale"), "Input(scale) should not be null.");
  PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Y")),
                 "Input(Y@GRAD) should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("SavedMean"),
                 "Input(SavedMean) should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("SavedVariance"),
                 "Input(SavedVariance) should not be null");

  // check output
  PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")), "");
  if (ctx->HasOutput(framework::GradVarName("Scale"))) {
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("Bias")),
                   "Output(Scale@GRAD) and Output(Bias@GRAD) should not be "
                   "null at same time");
  }
  const bool use_global_stats = ctx->Attrs().Get<bool>("use_global_stats");
  if (use_global_stats) {
    PADDLE_ENFORCE(!ctx->Attrs().Get<bool>("use_mkldnn"),
                   "Using global stats during training is not supported "
                   "in gradient op kernel of batch_norm_mkldnn_op now.");
  }

  const auto x_dims = ctx->GetInputDim("X");
  const DataLayout data_layout = framework::StringToDataLayout(
      ctx->Attrs().Get<std::string>("data_layout"));
  const int C = (data_layout == DataLayout::kNCHW ? x_dims[1]
                                                  : x_dims[x_dims.size() - 1]);

  ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
  if (ctx->HasOutput(framework::GradVarName("Scale"))) {
    ctx->SetOutputDim(framework::GradVarName("Scale"), {C});
    ctx->SetOutputDim(framework::GradVarName("Bias"), {C});
  }
}

framework::OpKernelType BatchNormGradOp::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
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

  return framework::OpKernelType(ctx.Input<Tensor>("X")->type(), ctx.GetPlace(),
                                 layout, library);
}

template <typename T>
class BatchNormGradKernel<platform::CPUDeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *x = ctx.Input<Tensor>("X");
    const auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *saved_mean = ctx.Input<Tensor>("SavedMean");
    // SavedVariance have been reverted in forward operator
    const auto *saved_inv_variance = ctx.Input<Tensor>("SavedVariance");
    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    const float epsilon = ctx.Attr<float>("epsilon");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);

    // Get the size for each dimension.
    // NCHW [batch_size, in_channels, in_height, in_width]
    const auto &x_dims = x->dims();
    PADDLE_ENFORCE(x_dims.size() >= 2 && x_dims.size() <= 5,
                   "The Input dim size should be between 2 and 5");
    const int N = x_dims[0];
    const int C =
        (data_layout == DataLayout::kNCHW ? x_dims[1]
                                          : x_dims[x_dims.size() - 1]);
    const int sample_size = x->numel() / N / C;

    // init output
    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    d_x->mutable_data<T>(ctx.GetPlace());

    const T *mean_data = saved_mean->data<T>();
    const T *inv_var_data = saved_inv_variance->data<T>();
    Tensor inv_var_tensor;
    if (use_global_stats) {
      const auto *running_mean = ctx.Input<Tensor>("Mean");
      const auto *running_variance = ctx.Input<Tensor>("Variance");
      mean_data = running_mean->data<T>();
      T *running_inv_var_data = inv_var_tensor.mutable_data<T>(ctx.GetPlace());
      EigenVectorArrayMap<T> inv_var_tmp(running_inv_var_data, C);
      ConstEigenVectorArrayMap<T> var_arr(running_variance->data<T>(), C);

      inv_var_tmp = (var_arr + epsilon).sqrt().inverse().eval();
      inv_var_data = running_inv_var_data;
    }

    ConstEigenVectorArrayMap<T> scale_arr(scale->data<T>(), C);
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

    switch (data_layout) {
      case DataLayout::kNCHW: {
        ConstEigenArrayMap<T> x_arr(x->data<T>(), sample_size, N * C);
        ConstEigenArrayMap<T> d_y_arr(d_y->data<T>(), sample_size, N * C);
        EigenArrayMap<T> d_x_arr(d_x->mutable_data<T>(ctx.GetPlace()),
                                 sample_size, N * C);
        d_x_arr.setZero();

        if (d_scale && d_bias) {
          for (int nc = 0; nc < N * C; ++nc) {
            int c = nc % C;
            d_bias_arr(c) += d_y_arr.col(nc).sum();
            d_scale_arr(c) += ((x_arr.col(nc) - mean_arr(c)) * inv_var_arr(c) *
                               d_y_arr.col(nc))
                                  .sum();
          }
        }
        if (!use_global_stats) {
          for (int nc = 0; nc < N * C; ++nc) {
            int c = nc % C;
            d_x_arr.col(nc) +=
                scale_inv_var_nhw(c) *
                (d_y_arr.col(nc) * N * sample_size - d_bias_arr(c) -
                 (x_arr.col(nc) - mean_arr[c]) * d_scale_arr(c) *
                     inv_var_arr(c));
          }
        } else {
          for (int nc = 0; nc < N * C; ++nc) {
            int c = nc % C;
            d_x_arr.col(nc) += scale_inv_var_nhw(c) * d_y_arr.col(nc);
          }
        }
        break;
      }
      case DataLayout::kNHWC: {
        ConstEigenArrayMap<T> x_arr(x->data<T>(), C, N * sample_size);
        ConstEigenArrayMap<T> d_y_arr(d_y->data<T>(), C, N * sample_size);
        EigenArrayMap<T> d_x_arr(d_x->mutable_data<T>(ctx.GetPlace()), C,
                                 N * sample_size);
        d_x_arr.setZero();

        const auto d_y_row_sum = d_y_arr.rowwise().sum();
        const auto x_minus_mean = x_arr.colwise() - mean_arr;
        const auto d_y_mul_x_minus_mean_row_sum =
            (d_y_arr * x_minus_mean).rowwise().sum();
        const auto inv_var_sqr = inv_var_arr * inv_var_arr;

        if (d_scale && d_bias) {
          for (int nhw = 0; nhw < N * sample_size; ++nhw) {
            d_bias_arr += d_y_arr.col(nhw);
            d_scale_arr +=
                (x_arr.col(nhw) - mean_arr) * inv_var_arr * d_y_arr.col(nhw);
          }
        }

        if (!use_global_stats) {
          for (int nhw = 0; nhw < N * sample_size; ++nhw) {
            d_x_arr.col(nhw) +=
                scale_inv_var_nhw *
                (d_y_arr.col(nhw) * N * sample_size - d_y_row_sum -
                 x_minus_mean.col(nhw) * inv_var_sqr *
                     d_y_mul_x_minus_mean_row_sum);
          }
        } else {
          for (int nhw = 0; nhw < N * sample_size; ++nhw) {
            d_x_arr.col(nhw) += scale_inv_var_nhw * d_y_arr.col(nhw);
          }
        }
        break;
      }
      default:
        PADDLE_THROW("Unknown storage order: %s", data_layout_str);
    }
  }
};

std::unique_ptr<framework::OpDesc> BatchNormGradMaker::Apply() const {
  auto *op = new framework::OpDesc();
  op->SetType(GradOpType());
  op->SetInput("X", Input("X"));
  op->SetInput(framework::GradVarName("Y"), OutputGrad("Y"));

  op->SetInput("Scale", Input("Scale"));
  op->SetInput("SavedMean", Output("SavedMean"));
  op->SetInput("SavedVariance", Output("SavedVariance"));

  // used when setting use_global_stats True during training
  if (boost::get<bool>(GetAttr("use_global_stats"))) {
    op->SetInput("Mean", Output("MeanOut"));
    op->SetInput("Variance", Output("VarianceOut"));
  }

  op->SetAttrMap(Attrs());

  op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
  op->SetOutput(framework::GradVarName("Scale"), InputGrad("Scale"));
  op->SetOutput(framework::GradVarName("Bias"), InputGrad("Bias"));

  return std::unique_ptr<framework::OpDesc>(op);
}

class BatchNormInplaceInToOut : public framework::InplaceInToOut {
 public:
  using InplaceInToOut::InplaceInToOut;

 protected:
  std::unordered_map<std::string, std::string> Apply(
      const framework::OpDesc &op_desc,
      framework::BlockDesc *block) const override {
    std::unordered_map<std::string, std::string> inplace_in_to_out = {
        {"Mean", "MeanOut"}, {"Variance", "VarianceOut"}, {"X", "Y"},
    };
    return inplace_in_to_out;
  }
};

class BatchNormGradInplaceInToOut : public framework::InplaceInToOut {
 public:
  using InplaceInToOut::InplaceInToOut;

 protected:
  std::unordered_map<std::string, std::string> Apply(
      const framework::OpDesc &op_desc,
      framework::BlockDesc *block) const override {
    std::unordered_map<std::string, std::string> inplace_in_to_out = {
        // Scale, Bias, SavedMean, SavedVariance shape is [batch_size, C]
        {framework::GradVarName("Y"), framework::GradVarName("X")},
        {"SavedMean", framework::GradVarName("Scale")},
        {"SavedVariance", framework::GradVarName("Bias")},
    };
    return inplace_in_to_out;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(batch_norm, ops::BatchNormOp, ops::BatchNormOpMaker,
                  ops::BatchNormOpInferVarType, ops::BatchNormGradMaker)
// ops::BatchNormInplaceInToOut);
REGISTER_OPERATOR(batch_norm_grad, ops::BatchNormGradOp)
//                  ops::BatchNormGradInplaceInToOut);

REGISTER_OP_CPU_KERNEL(
    batch_norm, ops::BatchNormKernel<paddle::platform::CPUDeviceContext, float>,
    ops::BatchNormKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    batch_norm_grad,
    ops::BatchNormGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::BatchNormGradKernel<paddle::platform::CPUDeviceContext, double>);
