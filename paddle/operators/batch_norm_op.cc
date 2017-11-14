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
    PADDLE_ENFORCE(ctx->HasInput("X"), "");
    PADDLE_ENFORCE(ctx->HasInput("Scale"), "");
    PADDLE_ENFORCE(ctx->HasInput("Bias"), "");
    PADDLE_ENFORCE(ctx->HasInput("Mean"), "");
    PADDLE_ENFORCE(ctx->HasInput("Variance"), "");
    PADDLE_ENFORCE(ctx->HasOutput("Y"), "");
    PADDLE_ENFORCE(ctx->HasOutput("MeanOut"), "");
    PADDLE_ENFORCE(ctx->HasOutput("VarianceOut"), "");
    PADDLE_ENFORCE(ctx->HasOutput("SavedMean"), "");
    PADDLE_ENFORCE(ctx->HasOutput("SavedVariance"), "");

    const float epsilon = ctx->Attrs().Get<float>("epsilon");
    PADDLE_ENFORCE_GE(epsilon, 0.0, "epsilon should be larger than 0");
    PADDLE_ENFORCE_LE(epsilon, 0.001, "epsilon should not be too large");

    // make sure Mean/MeanOut and Variance/VarianceOut share memory in Python
    PADDLE_ENFORCE_EQ(ctx->Inputs("Mean")[0], ctx->Outputs("MeanOut")[0],
                      "Mean and MeanOut should share the same memory");
    PADDLE_ENFORCE_EQ(ctx->Inputs("Variance")[0],
                      ctx->Outputs("VarianceOut")[0],
                      "Variance and VarianceOut should share the same memory");

    const auto x_dims = ctx->GetInputDim("X");
    const TensorFormat tensor_format =
        StringToTensorFormat(ctx->Attrs().Get<std::string>("tensor_format"));
    const int C =
        (tensor_format == TensorFormat::NCHW ? x_dims[1]
                                             : x_dims[x_dims.size() - 1]);

    PADDLE_ENFORCE(x_dims.size() >= 3 && x_dims.size() <= 5,
                   "Input X must have 3 to 5 dimensions.");

    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Scale").size(), 1UL);
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Scale")[0], C);
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Bias").size(), 1UL);
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Bias")[0], C);

    ctx->SetOutputDim("Y", x_dims);
    ctx->SetOutputDim("MeanOut", {C});
    ctx->SetOutputDim("VarianceOut", {C});
    ctx->SetOutputDim("SavedMean", {C});
    ctx->SetOutputDim("SavedVariance", {C});
  }
};

class BatchNormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  BatchNormOpMaker(framework::OpProto *proto,
                   framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddAttr<bool>("is_test", "").SetDefault(false);
    AddAttr<float>("momentum", "").SetDefault(0.9);
    AddAttr<float>("epsilon", "").SetDefault(1e-5);
    AddAttr<std::string>("tensor_format", "").SetDefault("NCHW");
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
1. NHWC `[batch, in_height, in_width, in_channels]`
2. NCHW `[batch, in_channels, in_height, in_width]`

)DOC");
  }
};

template <typename T>
class BatchNormKernel<platform::CPUPlace, T> : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const float epsilon = ctx.Attr<float>("epsilon");
    const float momentum = ctx.Attr<float>("momentum");
    const bool is_test = ctx.Attr<bool>("is_test");
    const std::string tensor_format_str =
        ctx.Attr<std::string>("tensor_format");
    const TensorFormat tensor_format = StringToTensorFormat(tensor_format_str);

    const auto *x = ctx.Input<Tensor>("X");
    const auto &x_dims = x->dims();
    PADDLE_ENFORCE(x_dims.size() >= 3 && x_dims.size() <= 5,
                   "The Input dim size should be between 3 and 5");
    const int N = x_dims[0];
    const int C =
        (tensor_format == TensorFormat::NCHW ? x_dims[1]
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

    if (!is_test) {
      // saved_xx is use just in this batch of data
      EigenVectorArrayMap<T> saved_mean_e(
          saved_mean->mutable_data<T>(ctx.GetPlace()), C);
      EigenVectorArrayMap<T> saved_variance_e(
          saved_variance->mutable_data<T>(ctx.GetPlace()), C);
      saved_mean_e.setZero();
      saved_variance_e.setZero();

      switch (tensor_format) {
        case TensorFormat::NCHW: {
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
        case TensorFormat::NHWC: {
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
          PADDLE_THROW("Unknown storage order: %s", tensor_format_str);
      }

      EigenVectorArrayMap<T> running_mean_arr(
          mean_out->mutable_data<T>(ctx.GetPlace()), C);
      EigenVectorArrayMap<T> running_var_arr(
          variance_out->mutable_data<T>(ctx.GetPlace()), C);
      running_mean_arr =
          running_mean_arr * momentum + saved_mean_e * (1. - momentum);
      running_var_arr =
          running_var_arr * momentum + saved_variance_e * (1. - momentum);
    }

    // use SavedMean and SavedVariance to do normalize
    Eigen::Array<T, Eigen::Dynamic, 1> inv_std(C);
    if (is_test) {
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
        is_test ? ctx.Input<Tensor>("Mean")->data<T>()
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

    switch (tensor_format) {
      case TensorFormat::NCHW: {
        EigenArrayMap<T> y_arr(y->mutable_data<T>(ctx.GetPlace()), sample_size,
                               N * C);
        ConstEigenArrayMap<T> x_arr(x->data<T>(), sample_size, N * C);
        for (int nc = 0; nc < N * C; ++nc) {
          y_arr.col(nc) = x_arr.col(nc) * new_scale(nc % C) + new_bias(nc % C);
        }
        break;
      }
      case TensorFormat::NHWC: {
        EigenArrayMap<T>(y->mutable_data<T>(ctx.GetPlace()), C,
                         N * sample_size) =
            (ConstEigenArrayMap<T>(x->data<T>(), C, N * sample_size).colwise() *
             new_scale)
                .colwise() +
            new_bias;
        break;
      }
      default:
        PADDLE_THROW("Unknown storage order: %d", tensor_format);
    }
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
    const TensorFormat tensor_format =
        StringToTensorFormat(ctx->Attrs().Get<std::string>("tensor_format"));
    const int C =
        (tensor_format == TensorFormat::NCHW ? x_dims[1]
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
    const auto *x = ctx.Input<Tensor>("X");
    const auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *saved_mean = ctx.Input<Tensor>("SavedMean");
    // SavedVariance have been reverted in forward operator
    const auto *saved_inv_variance = ctx.Input<Tensor>("SavedVariance");
    const std::string tensor_format_str =
        ctx.Attr<std::string>("tensor_format");
    const TensorFormat tensor_format = StringToTensorFormat(tensor_format_str);

    // Get the size for each dimension.
    // NCHW [batch_size, in_channels, in_height, in_width]
    const auto &x_dims = x->dims();
    PADDLE_ENFORCE(x_dims.size() >= 3 && x_dims.size() <= 5,
                   "The Input dim size should be between 3 and 5");
    const int N = x_dims[0];
    const int C =
        (tensor_format == TensorFormat::NCHW ? x_dims[1]
                                             : x_dims[x_dims.size() - 1]);
    const int sample_size = x->numel() / N / C;

    ConstEigenVectorArrayMap<T> scale_arr(scale->data<T>(), C);
    ConstEigenVectorArrayMap<T> mean_arr(saved_mean->data<T>(), C);
    ConstEigenVectorArrayMap<T> inv_var_arr(saved_inv_variance->data<T>(), C);

    // init output
    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    d_x->mutable_data<T>(ctx.GetPlace());
    d_scale->mutable_data<T>(ctx.GetPlace());
    d_bias->mutable_data<T>(ctx.GetPlace());

    // d_bias = np.sum(d_y, axis=0)
    // d_scale = np.sum((X - mean) / inv_std * dy, axis=0)
    // d_x = (1. / N) * scale * inv_var * (N * d_y - np.sum(d_y, axis=0)
    //   - (X - mean) * inv_var * inv_var * np.sum(d_y * (X - mean), axis=0))

    EigenVectorArrayMap<T> d_bias_arr(d_bias->mutable_data<T>(ctx.GetPlace()),
                                      C);
    EigenVectorArrayMap<T> d_scale_arr(d_scale->mutable_data<T>(ctx.GetPlace()),
                                       C);

    d_bias_arr.setZero();
    d_scale_arr.setZero();

    const auto scale_inv_var_nhw = scale_arr * inv_var_arr / (N * sample_size);

    switch (tensor_format) {
      case TensorFormat::NCHW: {
        ConstEigenArrayMap<T> x_arr(x->data<T>(), sample_size, N * C);
        ConstEigenArrayMap<T> d_y_arr(d_y->data<T>(), sample_size, N * C);
        EigenArrayMap<T> d_x_arr(d_x->mutable_data<T>(ctx.GetPlace()),
                                 sample_size, N * C);
        d_x_arr.setZero();

        for (int nc = 0; nc < N * C; ++nc) {
          int c = nc % C;
          d_bias_arr(c) += d_y_arr.col(nc).sum();
          d_scale_arr(c) +=
              ((x_arr.col(nc) - mean_arr(c)) * inv_var_arr(c) * d_y_arr.col(nc))
                  .sum();
        }
        for (int nc = 0; nc < N * C; ++nc) {
          int c = nc % C;
          d_x_arr.col(nc) +=
              scale_inv_var_nhw(c) *
              (d_y_arr.col(nc) * N * sample_size - d_bias_arr(c) -
               (x_arr.col(nc) - mean_arr[c]) * d_scale_arr(c) * inv_var_arr(c));
        }
        break;
      }
      case TensorFormat::NHWC: {
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
        for (int nhw = 0; nhw < N * sample_size; ++nhw) {
          d_bias_arr += d_y_arr.col(nhw);
          d_scale_arr +=
              (x_arr.col(nhw) - mean_arr) * inv_var_arr * d_y_arr.col(nhw);
          d_x_arr.col(nhw) +=
              scale_inv_var_nhw *
              (d_y_arr.col(nhw) * N * sample_size - d_y_row_sum -
               x_minus_mean.col(nhw) * inv_var_sqr *
                   d_y_mul_x_minus_mean_row_sum);
        }
        break;
      }
      default:
        PADDLE_THROW("Unknown storage order: %s", tensor_format_str);
    }
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
