// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <string>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

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

template <typename T, typename Context>
void DataNormGradKernel(const Context &dev_ctx,
                        const paddle::optional<DenseTensor> &scale_w_in,
                        const paddle::optional<DenseTensor> &bias_in,
                        const DenseTensor &x_in,
                        const DenseTensor &means_in,
                        const DenseTensor &scales_in,
                        const DenseTensor &out_grad,
                        float epsilon,
                        int slot_dim,
                        float summary_decay_rate,
                        bool enable_scale_and_shift,
                        const std::string &data_layout_in,
                        bool sync_stats,
                        DenseTensor *batch_size,
                        DenseTensor *batch_sum,
                        DenseTensor *batch_square_sum,
                        DenseTensor *scale_w_grad,
                        DenseTensor *bias_grad,
                        DenseTensor *x_grad,
                        DenseTensor *batch_size_grad,
                        DenseTensor *batch_sum_grad,
                        DenseTensor *batch_square_sum_grad) {
  const auto *x = &x_in;
  const auto *d_y = &out_grad;
  const auto *scales = &scales_in;
  const auto *means = &means_in;

  const std::string data_layout_str = data_layout_in;
  const DataLayout data_layout = common::StringToDataLayout(data_layout_str);

  // Get the size for each dimension.
  // NCHW [batch_size, in_channels, in_height, in_width]
  const auto &x_dims = x->dims();
  PADDLE_ENFORCE_EQ(
      x_dims.size(),
      2,
      common::errors::InvalidArgument("The Input dim size should be 2"));
  const int N = static_cast<int>(x_dims[0]);
  const int C = static_cast<int>(
      data_layout == DataLayout::kNCHW ? x_dims[1] : x_dims[x_dims.size() - 1]);
  // init output
  phi::DenseTensor *d_x = nullptr;
  if (x_grad != nullptr) {
    d_x = x_grad;
  }

  auto *d_batch_size = batch_size_grad;
  auto *d_batch_sum = batch_sum_grad;
  auto *d_batch_square_sum = batch_square_sum_grad;

  const T *mean_data = means->data<T>();
  const T *inv_var_data = scales->data<T>();
  ConstEigenVectorArrayMap<T> mean_arr(mean_data, C);
  ConstEigenVectorArrayMap<T> inv_var_arr(inv_var_data, C);

  T *d_batch_size_data = dev_ctx.template Alloc<T>(d_batch_size);
  T *d_batch_sum_data = dev_ctx.template Alloc<T>(d_batch_sum);
  T *d_batch_square_sum_data = dev_ctx.template Alloc<T>(d_batch_square_sum);
  EigenVectorArrayMap<T> d_batch_size_arr(d_batch_size_data, C);
  EigenVectorArrayMap<T> d_batch_sum_arr(d_batch_sum_data, C);
  EigenVectorArrayMap<T> d_batch_square_sum_arr(d_batch_square_sum_data, C);
  d_batch_size_arr.setZero();
  d_batch_sum_arr.setZero();
  d_batch_square_sum_arr.setZero();
  const T *x_data = x->data<T>();
  const T *means_data = means->data<T>();

  T min_precision = 1e-7f;
  switch (data_layout) {  // it's two dimensions, make no difference
    case DataLayout::kNCHW:
    case DataLayout::kNHWC: {
      ConstEigenVectorArrayMap<T> scales_arr(scales->data<T>(), C);
      ConstEigenVectorArrayMap<T> means_arr(means->data<T>(), C);
      ConstEigenArrayMap<T> x_arr(x->data<T>(), C, N);
      ConstEigenArrayMap<T> d_y_arr(d_y->data<T>(), C, N);
      if (d_x != nullptr) {
        EigenArrayMap<T> d_x_arr(dev_ctx.template Alloc<T>(d_x), C, N);
        d_x_arr.setZero();
        if (!enable_scale_and_shift) {
          for (int nc = 0; nc < N; ++nc) {
            d_x_arr.col(nc) = d_y_arr.col(nc) * scales_arr;
          }
        } else {
          const auto *scale_w = scale_w_in.get_ptr();
          auto *d_scale = scale_w_grad;
          auto *d_bias = bias_grad;
          ConstEigenVectorArrayMap<T> scale_arr(scale_w->data<T>(), C);
          T *d_bias_data = nullptr;
          T *d_scale_data = nullptr;

          d_bias_data = dev_ctx.template Alloc<T>(d_bias);
          d_scale_data = dev_ctx.template Alloc<T>(d_scale);

          EigenVectorArrayMap<T> d_bias_arr(d_bias_data, C);
          EigenVectorArrayMap<T> d_scale_arr(d_scale_data, C);
          phi::DenseTensor dy_sum;
          dy_sum.Resize({C});
          dev_ctx.template Alloc<T>(&dy_sum);
          EigenVectorArrayMap<T> dy_sum_arr(dy_sum.data<T>(), C);
          phi::DenseTensor dy_mul_x_sub_mean_mul_invstd_sum;
          dy_mul_x_sub_mean_mul_invstd_sum.Resize({C});
          dev_ctx.template Alloc<T>(&dy_mul_x_sub_mean_mul_invstd_sum);
          EigenVectorArrayMap<T> dy_mul_x_sub_mean_mul_invstd_sum_arr(
              dy_mul_x_sub_mean_mul_invstd_sum.data<T>(), C);

          dy_sum_arr.setZero();
          dy_mul_x_sub_mean_mul_invstd_sum_arr.setZero();

          if (slot_dim <= 0) {
            for (int n = 0; n < N; ++n) {
              dy_sum_arr += d_y_arr.col(n);
              dy_mul_x_sub_mean_mul_invstd_sum_arr +=
                  ((x_arr.col(n) - mean_arr) * inv_var_arr * d_y_arr.col(n));
            }
            if (d_scale && d_bias) {
              d_bias_arr = dy_sum_arr;
              d_scale_arr = dy_mul_x_sub_mean_mul_invstd_sum_arr;
            }
            for (int nc = 0; nc < N; ++nc) {
              d_x_arr.col(nc) = d_y_arr.col(nc) * scales_arr * scale_arr;
            }
          } else {
            int offset = 0;
            const int item_size = static_cast<int>(x->numel() / N);
            T *d_x_data = dev_ctx.template Alloc<T>(d_x);
            T *d_scale_data = dev_ctx.template Alloc<T>(d_scale);
            T *d_bias_data = dev_ctx.template Alloc<T>(d_bias);
            const T *dy_data = d_y->data<T>();
            const T *scales_data = scales->data<T>();
            const T *scale_w_data = scale_w->data<T>();
            const T *x_data = x->data<T>();
            for (int i = 0; i < item_size; i++) {
              d_bias_data[i] = 0;
              d_scale_data[i] = 0;
            }
            for (int k = 0; k < N; ++k) {
              for (int i = 0; i < item_size; i += slot_dim) {
                if (!(x_data[offset + i] > -min_precision &&
                      x_data[offset + i] < min_precision)) {
                  // show != 0
                  for (int j = i; j < i + slot_dim; ++j) {
                    d_x_data[offset + j] =
                        dy_data[offset + j] * scales_data[j] * scale_w_data[j];
                    d_bias_data[j] += dy_data[offset + j];
                    d_scale_data[j] += (x_data[offset + j] - mean_data[j]) *
                                       inv_var_data[j] * dy_data[offset + j];
                  }
                }
              }
              offset += item_size;
            }
          }
        }
      }

      if (slot_dim > 0 && N > 0) {
        // if slot_dim is set and batch size is larger than zero, we choose
        // to check if show number is zero, if so, skip update statistics.
        int offset = 0;
        const int item_size = static_cast<int>(x->numel() / N);
        for (int k = 0; k < N; ++k) {
          for (int i = 0; i < item_size; i += slot_dim) {
            if (!(x_data[offset + i] > -min_precision &&
                  x_data[offset + i] < min_precision)) {
              // show != 0
              for (int j = i; j < i + slot_dim; ++j) {
                d_batch_size_data[j] += 1;
                d_batch_sum_data[j] += x_data[offset + j];
                d_batch_square_sum_data[j] +=
                    (x_data[offset + j] - means_data[j]) *
                    (x_data[offset + j] - means_data[j]);
              }
            }
          }
          offset += item_size;
        }

        for (int i = 0; i < item_size; i += slot_dim) {
          for (int j = i; j < i + slot_dim; ++j) {
            if (d_batch_size_data[j] >= 1) {
              d_batch_sum_data[j] /= d_batch_size_data[j];
              d_batch_square_sum_data[j] =
                  d_batch_square_sum_data[j] / d_batch_size_data[j] +
                  d_batch_size_data[j] * epsilon;
              d_batch_size_data[j] = 1;
            }
          }
        }
      } else {
        // calculate data sum and square sum
        Eigen::Array<T, Eigen::Dynamic, 1> sample_sum(C);
        Eigen::Array<T, Eigen::Dynamic, 1> sample_square_sum(C);
        // calculate data sample sum and square sum
        sample_sum.setZero();
        sample_square_sum.setZero();
        for (int nc = 0; nc < N; ++nc) {
          sample_sum += x_arr.col(nc);
          sample_square_sum += (x_arr.col(nc) - means_arr).square();
        }
        // calculate gradient
        d_batch_size_arr.setConstant(N);
        d_batch_sum_arr = sample_sum;
        d_batch_square_sum_arr = sample_square_sum + d_batch_size_arr * epsilon;
      }
      break;
    }
    default:
      PADDLE_THROW(common::errors::InvalidArgument(
          "Unknown storage order: %s, please use NCHW or NHWC",
          data_layout_str));
  }
}

}  // namespace phi
PD_REGISTER_KERNEL(
    data_norm_grad, CPU, ALL_LAYOUT, phi::DataNormGradKernel, float, double) {}
