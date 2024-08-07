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
void DataNormKernel(const Context &dev_ctx,
                    const paddle::optional<DenseTensor> &scale_w_in,
                    const paddle::optional<DenseTensor> &bias_in,
                    const DenseTensor &x_in,
                    const DenseTensor &batch_size,
                    const DenseTensor &batch_sum,
                    const DenseTensor &batch_square_sum,
                    float epsilon,
                    int slot_dim,
                    float summary_decay_rate,
                    bool enable_scale_and_shift,
                    const std::string &data_layout_in,
                    bool sync_stats,
                    DenseTensor *out,
                    DenseTensor *means,
                    DenseTensor *scales) {
  const std::string data_layout_str = data_layout_in;
  const DataLayout data_layout = common::StringToDataLayout(data_layout_str);

  const auto *x = &x_in;
  const auto &x_dims = x->dims();
  PADDLE_ENFORCE_EQ(
      x_dims.size(),
      2,
      common::errors::InvalidArgument("The Input dim size should be 2"));
  const int N = static_cast<int>(x_dims[0]);
  const int C = static_cast<int>(
      data_layout == DataLayout::kNCHW ? x_dims[1] : x_dims[x_dims.size() - 1]);

  PADDLE_ENFORCE_LT(0,
                    N,
                    common::errors::InvalidArgument(
                        "The dims of Input(X) should be greater than 0."));
  PADDLE_ENFORCE_LT(0,
                    C,
                    common::errors::InvalidArgument(
                        "The dims of Input(X) should be greater than 0."));

  auto *y = out;
  auto *mean_out = means;

  // alloc memory
  T *y_data = dev_ctx.template Alloc<T>(y);

  ConstEigenVectorArrayMap<T> b_size_arr(batch_size.data<T>(), C);
  ConstEigenVectorArrayMap<T> b_sum_arr(batch_sum.data<T>(), C);
  ConstEigenVectorArrayMap<T> b_square_sum_arr(batch_square_sum.data<T>(), C);
  EigenVectorArrayMap<T> means_arr(dev_ctx.template Alloc<T>(mean_out), C);
  EigenVectorArrayMap<T> scales_arr(dev_ctx.template Alloc<T>(scales), C);
  means_arr = b_sum_arr / b_size_arr;
  scales_arr = (b_size_arr / b_square_sum_arr).sqrt();

  const T *means_data = mean_out->data<T>();
  const T *x_data = x->data<T>();

  const T *scales_data = scales->data<T>();
  T min_precision = 1e-7f;
  switch (data_layout) {
    case DataLayout::kNCHW:  // It's two dimensions, so make no difference
    case DataLayout::kNHWC: {
      // if slot_dim is set and batch size is larger than zero, we choose
      // to check if show number is zero, if so, skip normalization.
      if (slot_dim > 0 && N > 0 && (!enable_scale_and_shift)) {
        const int item_size = static_cast<int>(x->numel() / N);
        // location of show number in one embedding
        int offset = 0;
        for (int k = 0; k < N; ++k) {
          for (int i = 0; i < item_size; i += slot_dim) {
            if (x_data[offset + i] > -min_precision &&
                x_data[offset + i] < min_precision) {
              // show = 0
              memset(y_data + offset + i, 0, sizeof(T) * slot_dim);
            } else {
              for (int j = i; j < i + slot_dim; ++j) {
                y_data[offset + j] =
                    (x_data[offset + j] - means_data[j]) * scales_data[j];
              }
            }
          }

          offset += item_size;
        }
      } else {
        if (!enable_scale_and_shift && slot_dim <= 0) {
          EigenArrayMap<T>(y_data, C, N) =
              (ConstEigenArrayMap<T>(x->data<T>(), C, N).colwise() - means_arr)
                  .colwise() *
              scales_arr;
        } else if (enable_scale_and_shift && slot_dim <= 0) {
          const auto *scale_w = scale_w_in.get_ptr();
          const auto *bias = bias_in.get_ptr();
          ConstEigenVectorArrayMap<T> scale_w_arr(scale_w->data<T>(), C);
          ConstEigenVectorArrayMap<T> bias_arr(bias->data<T>(), C);

          Eigen::Array<T, Eigen::Dynamic, 1> new_scale =
              scales_arr * scale_w_arr;
          Eigen::Array<T, Eigen::Dynamic, 1> new_bias =
              bias_arr - means_arr * scales_arr * scale_w_arr;
          EigenArrayMap<T>(y_data, C, N) =
              (ConstEigenArrayMap<T>(x->data<T>(), C, N).colwise() * new_scale)
                  .colwise() +
              new_bias;

        } else {
          const int item_size = static_cast<int>(x->numel() / N);
          const auto *scale_w = scale_w_in.get_ptr();
          const auto *bias = bias_in.get_ptr();
          const T *scale_w_data = scale_w->data<T>();
          const T *bias_data = bias->data<T>();
          // location of show number in one embedding
          int offset = 0;
          for (int k = 0; k < N; ++k) {
            for (int i = 0; i < item_size; i += slot_dim) {
              if (x_data[offset + i] > -min_precision &&
                  x_data[offset + i] < min_precision) {
                // show = 0
                memset(y_data + offset + i, 0, sizeof(T) * slot_dim);
              } else {
                for (int j = i; j < i + slot_dim; ++j) {
                  y_data[offset + j] =
                      ((x_data[offset + j] - means_data[j]) * scales_data[j]) *
                          scale_w_data[j] +
                      bias_data[j];
                }
              }
            }  // end for i

            offset += item_size;
          }  // end for k
        }
      }
      break;
    }
    default:
      PADDLE_THROW(common::errors::InvalidArgument(
          "Unknown storage order: %d, please use NCHW or NHWC", data_layout));
  }
}

}  // namespace phi
PD_REGISTER_KERNEL(
    data_norm, CPU, ALL_LAYOUT, phi::DataNormKernel, float, double) {}
