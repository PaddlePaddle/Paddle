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

#pragma once
#include <string>

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DataLayout = framework::DataLayout;

template <typename D>
using DSizes = Eigen::DSizes<Eigen::DenseIndex, D>;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;

// template <typename T>
// using EigenArrayMap =
//     Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;
// template <typename T>
// using ConstEigenArrayMap =
//     Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;
// template <typename T>
// using EigenVectorArrayMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>;
// template <typename T>
// using ConstEigenVectorArrayMap =
//     Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>;

template <typename DeviceContext, typename T>
class BatchNormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const float epsilon = ctx.Attr<float>("epsilon");
    const float momentum = ctx.Attr<float>("momentum");
    const bool is_test = ctx.Attr<bool>("is_test");
    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);

    const auto *x = ctx.Input<Tensor>("X");
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");
    auto *y = ctx.Output<Tensor>("Y");
    y->mutable_data<T>(ctx.GetPlace());

    const auto &x_dims = x->dims();
    const int N = x_dims[0];
    const int C =
        (data_layout == DataLayout::kNCHW ? x_dims[1]
                                          : x_dims[x_dims.size() - 1]);
    const int sample_size = x->numel() / N / C;

    auto &dev_ctx = context.template device_context<DeviceContext>();
    auto *place = ctx.template device_context<DeviceContext>().eigen_device();

    if (is_test) {
      auto *est_mean = ctx.Input<Tensor>("Mean");
      auto *est_var = ctx.Input<Tensor>("Variance");

      auto mean_arr = EigenVector<T>::Flatten(*est_mean);
      auto var_arr = EigenVector<T>::Flatten(*est_var);
      auto scale_arr = EigenVector<T>::Flatten(*scale);
      auto bias_arr = EigenVector<T>::Flatten(*bias);
      auto inv_std = (var_arr + epsilon).rsqrt() * scale_arr;
      switch (data_layout) {
        case DataLayout::kNCHW: {
          auto y_arr = EigenTensor<T, 2>::From(
              *y, framework::make_ddim(sample_size, N * C));
          auto x_arr = EigenTensor<T, 2>::From(
              *x, framework::make_ddim(sample_size, N * C));
          DSizes<2> bcast_nchw(N * C, 1);
          y_arr.device(*place) = (x_arr - mean_arr.broadcast(bcast_nchw)) *
                                     ((var_arr + epsilon).rsqrt() * scale_arr)
                                         .eval()
                                         .broadcast(bcast_nchw) +
                                 bias_arr.broadcast(bcast_nchw);
          break;
        }
        case DataLayout::kNHWC: {
          DSizes<2> bcast_nchw(N * C, 1);
          break;
        }

        default:
          PADDLE_THROW("Unknown storage order: %d", data_layout);
      }
    } else {
      auto *mean_out = ctx.Output<Tensor>("MeanOut");
      auto *variance_out = ctx.Output<Tensor>("VarianceOut");
      // saved_xx is use just in this batch of data
      auto *saved_mean = ctx.Output<Tensor>("SavedMean");
      auto *saved_variance = ctx.Output<Tensor>("SavedVariance");
      mean_out->mutable_data<T>(ctx.GetPlace());
      variance_out->mutable_data<T>(ctx.GetPlace());
      saved_mean->mutable_data<T>(ctx.GetPlace());
      saved_variance->mutable_data<T>(ctx.GetPlace());

      auto saved_mean_e = EigenVector<T>::Flatten(*saved_mean);
      auto saved_variance_e = EigenVector<T>::Flatten(*saved_variance);
      math::SetConstant<DeviceContext, T> set_zero;
      set_zero(dev_ctx, saved_mean_e);
      set_zero(dev_ctx, saved_variance_e);

      switch (data_layout) {
        case DataLayout::kNCHW: {
          auto x_arr = EigenTensor<T, 2>::From(
              *x, framework::make_ddim(sample_size, N * C));
          auto x_arr = EigenTensor<T, 2>::From(
              *x, framework::make_ddim(sample_size, N * C));
          DSizes<2> bcast_nchw(N * C, 1);
          x_arr.sum

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
          auto x_arr = EigenTensor<T, 2>::From(
              *x, framework::make_ddim(C, N * sample_size));
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

      auto running_mean_arr = EigenVector<T>::Flatten(*mean_out);
      auto running_var_arr = EigenVector<T>::Flatten(*variance_out);
      running_mean_arr =
          running_mean_arr * momentum + saved_mean_e * (1. - momentum);
      running_var_arr =
          running_var_arr * momentum + saved_variance_e * (1. - momentum);
    }

    // use SavedMean and SavedVariance to do normalize
    Eigen::Array<T, Eigen::Dynamic, 1> inv_std(C);
    if (is_test) {
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

template <typename DeviceContext, typename T>
class BatchNormGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *x = ctx.Input<Tensor>("X");
    const auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *saved_mean = ctx.Input<Tensor>("SavedMean");
    // SavedVariance have been reverted in forward operator
    const auto *saved_inv_variance = ctx.Input<Tensor>("SavedVariance");
    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
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

    switch (data_layout) {
      case DataLayout::kNCHW: {
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
        PADDLE_THROW("Unknown storage order: %s", data_layout_str);
    }
  }
};

}  // namespace operators
}  // namespace paddle
