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

#pragma once
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
enum TensorFormat { NC = 0, NHWC, NCHW, NDHWC, NCDHW };

inline TensorFormat StringToTensorFormat(const std::string& str) {
  std::string format;
  std::transform(str.begin(), str.end(), std::back_inserter(format),
                 [](unsigned char c) -> unsigned char { return ::toupper(c); });
  if (format == "NC") {
    return TensorFormat::NC;
  } else if (format == "NHWC") {
    return TensorFormat::NHWC;
  } else if (format == "NCHW") {
    return TensorFormat::NCHW;
  } else if (format == "NDHWC") {
    return TensorFormat::NDHWC;
  } else if (format == "NCDHW") {
    return TensorFormat::NCDHW;
  } else {
    PADDLE_THROW("Unknown storage order string: %s", str);
  }
}

inline void ValidTensorRank(const int rank, const TensorFormat& tensor_format) {
  switch (tensor_format) {
    case TensorFormat::NC:
      PADDLE_ENFORCE(rank == 2,
                     "Input tensor rank should be 2 when format is NC.");
      break;
    case TensorFormat::NHWC:
    case TensorFormat::NCHW:
      PADDLE_ENFORCE(
          rank == 4,
          "Input tensor rank should be 4 when format is NHWC or NCHW.");
      break;
    case TensorFormat::NDHWC:
    case TensorFormat::NCDHW:
      PADDLE_ENFORCE(
          rank == 5,
          "Input tensor rank should be 5 when format is NDHWC or NCDHW.");
      break;
    default:
      PADDLE_THROW("Unknown storage order: %d", tensor_format);
  }
}

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
void BatchEstimate(const platform::Place place, const Tensor& x,
                   const TensorFormat& tensor_format, const int N, const int C,
                   const int sample_size, bool unbiased_variance,
                   EigenVectorArrayMap<T>& saved_mean_e,
                   EigenVectorArrayMap<T>& saved_variance_e) {
  switch (tensor_format) {
    case TensorFormat::NCHW:
    case TensorFormat::NCDHW: {
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
    case TensorFormat::NC:
    case TensorFormat::NHWC:
    case TensorFormat::NDHWC: {
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
      PADDLE_THROW("BatchEstimate not implemeted for storage order: %d",
                   tensor_format);
  }
  if (unbiased_variance && (N * sample_size) > 1) {
    saved_variance_e *= (N * sample_size) / (N * sample_size - 1);
  }
}

template <typename T>
void BatchNormalizeForward(
    const platform::Place place, const Tensor& x, const Tensor& scale,
    const Tensor& bias, const Tensor& mean_input, const Tensor& variance_input,
    const T epsilon, const T momentum, const std::string& tensor_format_str,
    bool unbiased_variance, bool is_test, Tensor* y, Tensor* mean_out,
    Tensor* variance_out, Tensor* saved_mean, Tensor* saved_variance) {
  const auto& x_dims = x.dims();
  auto tensor_format = StringToTensorFormat(tensor_format_str);
  ValidTensorRank(x_dims.size(), tensor_format);

  const int N = x_dims[0];
  const int C = (tensor_format == TensorFormat::NCHW ||
                         tensor_format == TensorFormat::NCDHW
                     ? x_dims[1]
                     : x_dims[x_dims.size() - 1]);
  const int sample_size = x.numel() / N / C;

  if (!is_test) {
    EigenVectorArrayMap<T> saved_mean_e(saved_mean->mutable_data<T>(place), C);
    EigenVectorArrayMap<T> saved_variance_e(
        saved_variance->mutable_data<T>(place), C);
    saved_mean_e.setZero();
    saved_variance_e.setZero();
    // saved_xx are estimated using current batch
    BatchEstimate<T>(place, x, tensor_format, N, C, sample_size,
                     unbiased_variance, saved_mean_e, saved_variance_e);
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
    case TensorFormat::NCHW:
    case TensorFormat::NCDHW: {
      EigenArrayMap<T> y_arr(y->mutable_data<T>(place), sample_size, N * C);
      ConstEigenArrayMap<T> x_arr(x.data<T>(), sample_size, N * C);
      for (int nc = 0; nc < N * C; ++nc) {
        y_arr.col(nc) = x_arr.col(nc) * new_scale(nc % C) + new_bias(nc % C);
      }
      break;
    }
    case TensorFormat::NC:
    case TensorFormat::NHWC:
    case TensorFormat::NDHWC: {
      EigenArrayMap<T>(y->mutable_data<T>(place), C, N * sample_size) =
          (ConstEigenArrayMap<T>(x.data<T>(), C, N * sample_size).colwise() *
           new_scale)
              .colwise() +
          new_bias;
      break;
    }
    default:
      PADDLE_THROW("Unknown storage order: %s", tensor_format_str);
  }
}

template <typename T>
void BatchNormalizeBackward(const platform::Place place, const Tensor& d_y,
                            const Tensor& x, const Tensor& scale,
                            const Tensor& saved_mean,
                            const Tensor& saved_inv_variance,
                            const std::string& tensor_format_str, Tensor* d_x,
                            Tensor* d_scale, Tensor* d_bias) {
  const auto& x_dims = x.dims();
  auto tensor_format = StringToTensorFormat(tensor_format_str);
  ValidTensorRank(x_dims.size(), tensor_format);
  const int N = x_dims[0];
  const int C = (tensor_format == TensorFormat::NCHW ||
                         tensor_format == TensorFormat::NCDHW
                     ? x_dims[1]
                     : x_dims[x_dims.size() - 1]);
  const int sample_size = x.numel() / N / C;

  ConstEigenVectorArrayMap<T> scale_arr(scale.data<T>(), C);
  ConstEigenVectorArrayMap<T> mean_arr(saved_mean.data<T>(), C);
  ConstEigenVectorArrayMap<T> inv_var_arr(saved_inv_variance.data<T>(), C);

  // d_bias = np.sum(d_y, axis=0)
  // d_scale = np.sum((X - mean) / inv_std * dy, axis=0)
  // d_x = (1. / N) * scale * inv_var * (N * d_y - np.sum(d_y, axis=0)
  //   - (X - mean) * inv_var * inv_var * np.sum(d_y * (X - mean), axis=0))
  EigenVectorArrayMap<T> d_bias_arr(d_bias->mutable_data<T>(place), C);
  EigenVectorArrayMap<T> d_scale_arr(d_scale->mutable_data<T>(place), C);
  d_bias_arr.setZero();
  d_scale_arr.setZero();

  const auto scale_inv_var_nhw = scale_arr * inv_var_arr / (N * sample_size);

  switch (tensor_format) {
    case TensorFormat::NCHW:
    case TensorFormat::NCDHW: {
      ConstEigenArrayMap<T> x_arr(x.data<T>(), sample_size, N * C);
      ConstEigenArrayMap<T> d_y_arr(d_y.data<T>(), sample_size, N * C);
      EigenArrayMap<T> d_x_arr(d_x->mutable_data<T>(place), sample_size, N * C);
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
    case TensorFormat::NC:
    case TensorFormat::NHWC:
    case TensorFormat::NDHWC: {
      ConstEigenArrayMap<T> x_arr(x.data<T>(), C, N * sample_size);
      ConstEigenArrayMap<T> d_y_arr(d_y.data<T>(), C, N * sample_size);
      EigenArrayMap<T> d_x_arr(d_x->mutable_data<T>(place), C, N * sample_size);
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
        d_x_arr.col(nhw) += scale_inv_var_nhw *
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

}  // namespace operators
}  // namespace paddle
