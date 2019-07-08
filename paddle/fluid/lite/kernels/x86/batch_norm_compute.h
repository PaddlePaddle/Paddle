// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <random>
#include <string>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

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
class BatchNormCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::BatchNormParam;
  void Run() override {
    auto &param = *param_.get_mutable<operators::BatchNormParam>();
    bool global_stats = param.is_test || param.use_global_stats;

    const auto *x = param.x;
    const auto &x_dims = x->dims();
    CHECK(x_dims.size() >= 2 && x_dims.size() <= 5);
    const int N = x_dims[0];
    const int C = param.data_layout == DATALAYOUT(kNCHW)
                      ? x_dims[1]
                      : x_dims[x_dims.size() - 1];
    const int sample_size = x->dims().production() / N / C;

    // alloc memory
    param.y->template mutable_data<T>();
    if (!param.is_test) {
      param.mean_out->template mutable_data<T>();
      param.variance_out->template mutable_data<T>();
      param.saved_mean->template mutable_data<T>();
      param.saved_variance->template mutable_data<T>();
    }
    if (!global_stats) {
      // saved_xx is use just in this batch of data
      EigenVectorArrayMap<T> saved_mean_e(param.saved_mean->mutable_data<T>(),
                                          C);
      EigenVectorArrayMap<T> saved_variance_e(
          param.saved_variance->mutable_data<T>(), C);
      saved_mean_e.setZero();
      saved_variance_e.setZero();

      EigenVectorArrayMap<T> running_mean_arr(param.mean_out->mutable_data<T>(),
                                              C);
      EigenVectorArrayMap<T> running_var_arr(
          param.variance_out->mutable_data<T>(), C);

      if ((N * sample_size) == 1) {
        LOG(WARNING) << "Only 1 element in normalization dimension, "
                     << "we skip the batch norm calculation, let y = x.";
        framework::TensorCopy(x->raw_tensor(), platform::CPUPlace(),
                              &param.y->raw_tensor());
        return;
      }

      switch (param.data_layout) {
        case DATALAYOUT(kNCHW): {
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
        default:
          LOG(FATAL) << "Unknown storage order: "
                     << DataLayoutToStr(param.data_layout);
          break;
      }
      running_mean_arr = running_mean_arr * param.momentum +
                         saved_mean_e * (1. - param.momentum);
      running_var_arr = running_var_arr * param.momentum +
                        saved_variance_e * (1. - param.momentum);
    }

    // use SavedMean and SavedVariance to do normalize
    Eigen::Array<T, Eigen::Dynamic, 1> inv_std(C);
    if (global_stats) {
      ConstEigenVectorArrayMap<T> var_arr(param.variance->data<T>(), C);
      inv_std = (var_arr + param.epsilon).sqrt().inverse();
    } else {
      EigenVectorArrayMap<T> saved_inv_std(
          param.saved_variance->mutable_data<T>(), C);
      // inverse SavedVariance first, gradient will use it too.
      saved_inv_std = (saved_inv_std + param.epsilon).inverse().sqrt();
      inv_std = saved_inv_std;
    }

    ConstEigenVectorArrayMap<T> mean_arr(
        global_stats ? param.mean->data<T>() : param.saved_mean->data<T>(), C);

    //   ((x - est_mean) * (inv_var) * scale + bias
    //   formula transform ====>
    //   (x * inv_var * scale) + (bias - est_mean * inv_var * scale)

    ConstEigenVectorArrayMap<T> scale_arr(param.scale->data<T>(), C);
    ConstEigenVectorArrayMap<T> bias_arr(param.bias->data<T>(), C);
    Eigen::Array<T, Eigen::Dynamic, 1> new_scale = inv_std * scale_arr;
    Eigen::Array<T, Eigen::Dynamic, 1> new_bias =
        bias_arr - mean_arr * inv_std * scale_arr;

    switch (param.data_layout) {
      case DATALAYOUT(kNCHW): {
        EigenArrayMap<T> y_arr(param.y->mutable_data<T>(), sample_size, N * C);
        ConstEigenArrayMap<T> x_arr(x->data<T>(), sample_size, N * C);
        for (int nc = 0; nc < N * C; ++nc) {
          y_arr.col(nc) = x_arr.col(nc) * new_scale(nc % C) + new_bias(nc % C);
        }
        break;
      }
      default:
        LOG(FATAL) << "Unknown storage order: "
                   << DataLayoutToStr(param.data_layout);
        break;
    }
  }
  virtual ~BatchNormCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
