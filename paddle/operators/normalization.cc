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

#include "paddle/operators/normalization.h"

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

template <typename T>
void BatchNormalizeForward(const platform::Place place, const Tensor& x,
                           const Tensor& scale, const Tensor& bias,
                           const Tensor& mean_input,
                           const Tensor& variance_input, const T epsilon,
                           const T momentum, const TensorFormat& tensor_format,
                           bool is_test, Tensor* y, Tensor* mean_out,
                           Tensor* variance_out, Tensor* saved_mean,
                           Tensor* saved_variance) {
  const auto& x_dims = x.dims();
  PADDLE_ENFORCE(x_dims.size() >= 3 && x_dims.size() <= 5,
                 "The Input dim size should be between 3 and 5");
  const int N = x_dims[0];
  const int C =
      (tensor_format == TensorFormat::NCHW ? x_dims[1]
                                           : x_dims[x_dims.size() - 1]);
  const int sample_size = x.numel() / N / C;

  if (!is_test) {
    // saved_xx are estimated using current batch
    EigenVectorArrayMap<T> saved_mean_e(saved_mean->mutable_data<T>(place), C);
    EigenVectorArrayMap<T> saved_variance_e(
        saved_variance->mutable_data<T>(place), C);
    saved_mean_e.setZero();
    saved_variance_e.setZero();

    switch (tensor_format) {
      case TensorFormat::NCHW: {
        ConstEigenArrayMap<T> x_arr(x.data<T>(), sample_size, N * C);
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
        ConstEigenArrayMap<T> x_arr(x.data<T>(), C, N * sample_size);
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

    EigenVectorArrayMap<T> running_mean_arr(mean_out->mutable_data<T>(place),
                                            C);
    EigenVectorArrayMap<T> running_var_arr(variance_out->mutable_data<T>(place),
                                           C);
    running_mean_arr =
        running_mean_arr * momentum + saved_mean_e * (1. - momentum);
    running_var_arr =
        running_var_arr * momentum + saved_variance_e * (1. - momentum);
  }

  // use SavedMean and SavedVariance to do normalize
  Eigen::Array<T, Eigen::Dynamic, 1> inv_std(C);
  if (is_test) {
    ConstEigenVectorArrayMap<T> var_arr(variance_input.data<T>(), C);
    inv_std = (var_arr + epsilon).sqrt().inverse();
  } else {
    EigenVectorArrayMap<T> saved_inv_std(saved_variance->data<T>(), C);
    // inverse SavedVariance first, gradient will use it too.
    saved_inv_std = (saved_inv_std + epsilon).inverse().sqrt();
    inv_std = saved_inv_std;
  }
  ConstEigenVectorArrayMap<T> mean_arr(
      is_test ? mean_input.data<T>() : saved_mean->data<T>(), C);

  //   ((x - est_mean) * (inv_var) * scale + bias
  //   formula transform ====>
  //   (x * inv_var * scale) + (bias - est_mean * inv_var * scale)
  ConstEigenVectorArrayMap<T> scale_arr(scale.data<T>(), C);
  ConstEigenVectorArrayMap<T> bias_arr(bias.data<T>(), C);
  Eigen::Array<T, Eigen::Dynamic, 1> new_scale = inv_std * scale_arr;
  Eigen::Array<T, Eigen::Dynamic, 1> new_bias =
      bias_arr - mean_arr * inv_std * scale_arr;

  switch (tensor_format) {
    case TensorFormat::NCHW: {
      EigenArrayMap<T> y_arr(y->mutable_data<T>(place), sample_size, N * C);
      ConstEigenArrayMap<T> x_arr(x->data<T>(), sample_size, N * C);
      for (int nc = 0; nc < N * C; ++nc) {
        y_arr.col(nc) = x_arr.col(nc) * new_scale(nc % C) + new_bias(nc % C);
      }
      break;
    }
    case TensorFormat::NHWC: {
      EigenArrayMap<T>(y->mutable_data<T>(place), C, N * sample_size) =
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

}  // namespace operators
}  // namespace paddle
