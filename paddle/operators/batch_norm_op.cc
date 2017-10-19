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
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

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

    // make sure Mean/MeanOut and Variance/VarianceOut share memory in Python
    PADDLE_ENFORCE_EQ(ctx->Inputs("Mean")[0], ctx->Outputs("MeanOut")[0],
                      "Mean and MeanOut should share the same memory");
    PADDLE_ENFORCE_EQ(ctx->Inputs("Variance")[0],
                      ctx->Outputs("VarianceOut")[0],
                      "Variance and VarianceOut should share the same memory");

    const auto x_dims = ctx->GetInputDim("X");
    const int C = x_dims[1];  // channel num

    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Scale").size(), 1UL);
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Scale")[0], C);
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Bias").size(), 1UL);
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Bias")[0], C);

    ctx->SetOutputDim("Out", x_dims);
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
    AddInput("X", "The input 4-dimensional tensor");
    AddInput("Scale", "The second input of mul op");
    AddInput("Bias",
             "The bias as a 1-dimensional "
             "tensor of size C to be applied to the output");
    AddInput("Mean",
             "The running mean (training) or the "
             "estimated mean (testing)");
    AddInput("Variance",
             "The running variance (training) "
             "or the estimated");
    AddOutput("Y", "result after normalized");
    AddOutput("MeanOut",
              "The running mean (training) or the "
              "estimated mean (testing)");
    AddOutput("VarianceOut",
              "The running variance (training) "
              "or the estimated");
    AddOutput("SavedMean", "Local Mean");
    AddOutput("SavedVariance", "Local Variance");
    AddComment(R"DOC(
https://arxiv.org/pdf/1502.03167.pdf

NHWC `[batch, in_height, in_width, in_channels]`
NCHW `[batch, in_channels, in_height, in_width]`

we choose NCHW as the order.

)DOC");
  }
};

// BatchNormKernel for CPU, now only support NCHW data format
template <typename T>
class BatchNormKernel<platform::CPUPlace, T> : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const T momentum = static_cast<T>(ctx.Attr<float>("momentum"));
    const bool is_test = ctx.Attr<bool>("is_test");
    const float epsilon = ctx.Attr<float>("epsilon");

    const auto *x = ctx.Input<Tensor>("X");

    // Get the size for each dimension.
    // NCHW [batch_size, in_channels, in_height, in_width]
    const auto &x_dims = x->dims();
    PADDLE_ENFORCE(x_dims.size() >= 3 && x_dims.size() <= 5,
                   "The Input dim size should be between 3 and 5");
    const int N = x_dims[0];
    const int C = x_dims[1];
    const int H = x_dims[2];
    const int W = x_dims.size() > 3 ? x_dims[3] : 1;
    const int D = x_dims.size() > 4 ? x_dims[4] : 1;

    const int sample_size = H * W * D;

    // const auto& place = ctx.GetEigenDevice<Place>();
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

      ConstEigenArrayMap<T> X_arr(x->data<T>(), sample_size, N * C);
      for (int nc = 0; nc < N * C; ++nc) {
        saved_mean_e(nc % C) += X_arr.col(nc).sum();
      }
      saved_mean_e /= N * sample_size;
      for (int nc = 0; nc < N * C; ++nc) {
        saved_variance_e(nc % C) +=
            (X_arr.col(nc) - saved_variance_e(nc % C)).matrix().squaredNorm();
      }
      saved_variance_e /= N * sample_size;

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
    EigenArrayMap<T> Y_arr(y->mutable_data<T>(ctx.GetPlace()), sample_size,
                           N * C);
    ConstEigenArrayMap<T> X_arr(x->data<T>(), sample_size, N * C);
    for (int nc = 0; nc < N * C; ++nc) {
      Y_arr.col(nc) = X_arr.col(nc) * new_scale(nc % C) + new_bias(nc % C);
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
    const int C = x_dims[1];  // channel num

    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    ctx->SetOutputDim(framework::GradVarName("Scale"), {C});
    ctx->SetOutputDim(framework::GradVarName("Bias"), {C});
  }
};

// BatchNormKernel for CPU, now only support NCHW data format
template <typename T>
class BatchNormGradKernel<platform::CPUPlace, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *x = ctx.Input<Tensor>("X");
    const auto *dY = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *saved_mean = ctx.Input<Tensor>("SavedMean");
    // SavedVariance have been reverted in forward operator
    const auto *saved_inv_variance = ctx.Input<Tensor>("SavedVariance");

    // Get the size for each dimension.
    // NCHW [batch_size, in_channels, in_height, in_width]
    const auto &x_dims = x->dims();
    PADDLE_ENFORCE(x_dims.size() >= 3 && x_dims.size() <= 5,
                   "The Input dim size should be between 3 and 5");
    const int N = x_dims[0];
    const int C = x_dims[1];
    const int H = x_dims[2];
    const int W = x_dims.size() > 3 ? x_dims[3] : 1;
    const int D = x_dims.size() > 4 ? x_dims[4] : 1;

    const int sample_size = H * W * D;

    ConstEigenVectorArrayMap<T> scale_arr(scale->data<T>(), C);
    ConstEigenVectorArrayMap<T> mean_arr(saved_mean->data<T>(), C);
    ConstEigenVectorArrayMap<T> inv_var_arr(saved_inv_variance->data<T>(), C);

    // init output
    auto *dX = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *dScale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *dBias = ctx.Output<Tensor>(framework::GradVarName("Biase"));

    dX->mutable_data<T>(ctx.GetPlace());
    dScale->mutable_data<T>(ctx.GetPlace());
    dBias->mutable_data<T>(ctx.GetPlace());

    // dBias = np.sum(dY, axis=0)
    // dScale = np.sum((X - mean) / inv_std * dy, axis=0)
    // dX = (1. / N) * scale * inv_var * (N * dY - np.sum(dY, axis=0)
    //   - (X - mean) * inv_var * inv_var * np.sum(dY * (X - mean), axis=0))

    EigenVectorArrayMap<T> dBias_arr(dBias->mutable_data<T>(ctx.GetPlace()), C);
    EigenVectorArrayMap<T> dScale_arr(dScale->mutable_data<T>(ctx.GetPlace()),
                                      C);

    dBias_arr.setZero();
    dScale_arr.setZero();

    const auto scaleInvVarNHW = scale_arr * inv_var_arr / (N * sample_size);

    ConstEigenArrayMap<T> X_arr(x->data<T>(), sample_size, N * C);
    ConstEigenArrayMap<T> dY_arr(dY->data<T>(), sample_size, N * C);
    EigenArrayMap<T> dX_arr(dX->mutable_data<float>(ctx.GetPlace()),
                            sample_size, N * C);
    dX_arr.setZero();

    for (int nc = 0; nc < N * C; ++nc) {
      int c = nc % C;
      dBias_arr(c) += dY_arr.col(nc).sum();
      dScale_arr(c) +=
          ((X_arr.col(nc) - mean_arr(c)) * inv_var_arr(c) * dY_arr.col(nc))
              .sum();
    }
    for (int nc = 0; nc < N * C; ++nc) {
      int c = nc % C;
      dX_arr.col(nc) +=
          scaleInvVarNHW(c) *
          (dY_arr.col(nc) * N * sample_size - dBias_arr(c) -
           (X_arr.col(nc) - mean_arr[c]) * dScale_arr(c) * inv_var_arr(c));
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
