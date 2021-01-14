/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <memory>
#include <string>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
static void nll_loss_1D(T* out_data, T* total_weight_data, const T* x_data,
                        const int64_t* label_data, const T* weight_data,
                        const int64_t batch_size, const int64_t n_classes,
                        const std::string reduction,
                        const int64_t ignore_index) {
  if (reduction == "none") {
    for (int64_t i = 0; i < batch_size; ++i) {
      const auto cur_label = label_data[i];
      if (cur_label == ignore_index) {
        out_data[i] = 0;
        continue;
      }
      PADDLE_ENFORCE_EQ(cur_label >= 0 && cur_label < n_classes, true,
                        platform::errors::InvalidArgument(
                            "label should not be out of bounds."));

      const auto cur_weight =
          weight_data ? weight_data[cur_label] : static_cast<T>(1);
      out_data[i] = -x_data[i * n_classes + cur_label] * cur_weight;
    }
    return;
  }

  T output_val = 0;
  T total_weight_val = 0;

  for (int64_t i = 0; i < batch_size; i++) {
    const auto cur_label = label_data[i];
    if (cur_label == ignore_index) {
      out_data[i] = 0;
      continue;
    }
    PADDLE_ENFORCE_EQ(cur_label >= 0 && cur_label < n_classes, true,
                      platform::errors::InvalidArgument(
                          "label should not be out of bounds."));

    const auto cur_weight =
        weight_data ? weight_data[cur_label] : static_cast<T>(1);
    total_weight_val += cur_weight;
    output_val -= x_data[i * n_classes + cur_label] * cur_weight;
  }
  if (reduction == "mean" && total_weight_val != 0) {
    output_val /= total_weight_val;
  }
  *out_data = output_val;
  *total_weight_data = total_weight_val;
}

template <typename T>
static void nll_loss_2D(T* out_data, T* total_weight_data, const T* x_data,
                        const int64_t* label_data, const T* weight_data,
                        const int64_t batch_size, const int64_t n_classes,
                        const int64_t in_dim2, const int64_t in_dim3,
                        const std::string reduction,
                        const int64_t ignore_index) {
  const auto map_size = in_dim2 * in_dim3;
  const auto sample_size = n_classes * map_size;
  if (reduction == "none") {
    for (int i = 0; i < batch_size; i++) {
      for (int h = 0; h < in_dim2; h++) {
        for (int w = 0; w < in_dim3; w++) {
          const auto index = i * map_size + h * in_dim3 + w;
          const auto cur_label = label_data[index];
          if (cur_label == ignore_index) {
            out_data[index] = 0;
            continue;
          }
          PADDLE_ENFORCE_EQ(cur_label >= 0 && cur_label < n_classes, true,
                            platform::errors::InvalidArgument(
                                "label should not be out of bounds."));
          const auto cur_weight =
              weight_data ? weight_data[cur_label] : static_cast<T>(1);
          out_data[index] = -x_data[i * sample_size + cur_label * map_size +
                                    h * in_dim3 + w] *
                            cur_weight;
        }
      }
    }
    return;
  }

  T output_val = 0;
  T total_weight_val = 0;

  for (int i = 0; i < batch_size; i++) {
    for (int h = 0; h < in_dim2; h++) {
      for (int w = 0; w < in_dim3; w++) {
        const auto index = i * map_size + h * in_dim3 + w;
        const auto cur_label = label_data[index];
        if (cur_label == ignore_index) {
          out_data[index] = 0;
          continue;
        }
        PADDLE_ENFORCE_EQ(cur_label >= 0 && cur_label < n_classes, true,
                          platform::errors::InvalidArgument(
                              "label should not be out of bounds."));
        const auto cur_weight =
            weight_data ? weight_data[cur_label] : static_cast<T>(1);
        total_weight_val += cur_weight;
        output_val -=
            x_data[i * sample_size + cur_label * map_size + h * in_dim3 + w] *
            cur_weight;
      }
    }
  }

  if (reduction == "mean" && total_weight_val != 0) {
    output_val /= total_weight_val;
  }
  *out_data = output_val;
  *total_weight_data = total_weight_val;
}

template <typename DeviceContext, typename T>
class NLLLossOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* labels = ctx.Input<Tensor>("Label");
    auto* weight = ctx.Input<Tensor>("Weight");
    auto* out = ctx.Output<Tensor>("Out");
    auto* total_weight = ctx.Output<Tensor>("Total_weight");
    auto reduction = ctx.Attr<std::string>("reduction");
    auto ignore_index = ctx.Attr<int64_t>("ignore_index");

    auto x_data = x->data<T>();
    auto label_data = labels->data<int64_t>();
    auto weight_data = weight ? weight->data<T>() : nullptr;
    auto out_data = out->mutable_data<T>(ctx.GetPlace());
    auto total_weight_data = total_weight->mutable_data<T>(ctx.GetPlace());
    *total_weight_data = 0;

    auto x_dims = x->dims();
    const auto batch_size = x_dims[0];
    const auto n_classes = x_dims[1];

    if (x_dims.size() == 2) {
      nll_loss_1D<T>(out_data, total_weight_data, x_data, label_data,
                     weight_data, batch_size, n_classes, reduction,
                     ignore_index);
    } else if (x_dims.size() == 4) {
      const auto in_dim2 = x_dims[2];
      const auto in_dim3 = x_dims[3];
      nll_loss_2D<T>(out_data, total_weight_data, x_data, label_data,
                     weight_data, batch_size, n_classes, in_dim2, in_dim3,
                     reduction, ignore_index);
    }
  }
};

template <typename T>
static void nll_loss_grad_1D(T* dx_data, const T* dout_data,
                             const int64_t* label_data, const T* weight_data,
                             const T* total_weight_data,
                             const int64_t batch_size, const int64_t n_classes,
                             const std::string reduction,
                             const int64_t ignore_index) {
  if (reduction == "none") {
    for (int i = 0; i < batch_size; i++) {
      const auto cur_label = label_data[i];
      if (cur_label == ignore_index) {
        continue;
      }
      const auto cur_weight =
          weight_data ? weight_data[cur_label] : static_cast<T>(1);
      dx_data[i * n_classes + cur_label] = -dout_data[i] * cur_weight;
    }
    return;
  }

  const T dout_val = *dout_data;
  const T total_weight_val = *total_weight_data;
  for (int i = 0; i < batch_size; i++) {
    const auto cur_label = label_data[i];
    if (cur_label == ignore_index) {
      continue;
    }
    const auto cur_weight =
        weight_data ? weight_data[cur_label] : static_cast<T>(1);
    dx_data[i * n_classes + cur_label] = -dout_val * cur_weight;
    if (reduction == "mean") {
      dx_data[i * n_classes + cur_label] /= total_weight_val;
    }
  }
}

template <typename T>
static void nll_loss_grad_2D(T* dx_data, const T* dout_data,
                             const int64_t* label_data, const T* weight_data,
                             const T* total_weight_data,
                             const int64_t batch_size, const int64_t n_classes,
                             const int64_t in_dim2, const int64_t in_dim3,
                             const std::string reduction,
                             const int64_t ignore_index) {
  const auto map_size = in_dim2 * in_dim3;
  const auto sample_size = n_classes * map_size;

  if (reduction == "none") {
    for (int i = 0; i < batch_size; i++) {
      for (int h = 0; h < in_dim2; h++) {
        for (int w = 0; w < in_dim3; w++) {
          const auto index = i * map_size + h * in_dim3 + w;
          const auto cur_label = label_data[index];
          if (cur_label == ignore_index) {
            continue;
          }
          const auto cur_weight =
              weight_data ? weight_data[cur_label] : static_cast<T>(1);
          dx_data[i * sample_size + cur_label * map_size + h * in_dim3 + w] =
              -cur_weight * dout_data[index];
        }
      }
    }
    return;
  }

  const T dout_val = *dout_data;
  const T total_weight_val = *total_weight_data;
  for (int i = 0; i < batch_size; i++) {
    for (int h = 0; h < in_dim2; h++) {
      for (int w = 0; w < in_dim3; w++) {
        const auto index = i * map_size + h * in_dim3 + w;
        const auto cur_label = label_data[index];
        if (cur_label == ignore_index) {
          continue;
        }
        const auto cur_weight =
            weight_data ? weight_data[cur_label] : static_cast<T>(1);
        const auto dx_index =
            i * sample_size + cur_label * map_size + h * in_dim3 + w;
        dx_data[dx_index] = -dout_val * cur_weight;
        if (reduction == "mean") {
          dx_data[dx_index] /= total_weight_val;
        }
      }
    }
  }
}

template <typename DeviceContext, typename T>
class NLLLossGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* labels = ctx.Input<Tensor>("Label");
    auto* weight = ctx.Input<Tensor>("Weight");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* total_weight = ctx.Input<Tensor>("Total_weight");
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto ignore_index = ctx.Attr<int64_t>("ignore_index");
    auto reduction = ctx.Attr<std::string>("reduction");

    auto dx_data = dx->mutable_data<T>(ctx.GetPlace());
    auto dout_data = dout->data<T>();
    auto label_data = labels->data<int64_t>();
    auto weight_data = weight ? weight->data<T>() : nullptr;
    auto total_weight_data = total_weight->data<T>();
    memset(dx_data, 0, dx->numel() * sizeof(T));

    const auto x_dims = x->dims();
    const auto batch_size = x_dims[0];
    const auto n_classes = x_dims[1];

    if (x_dims.size() == 2) {
      nll_loss_grad_1D(dx_data, dout_data, label_data, weight_data,
                       total_weight_data, batch_size, n_classes, reduction,
                       ignore_index);
    } else if (x_dims.size() == 4) {
      const auto in_dim2 = x_dims[2];
      const auto in_dim3 = x_dims[3];
      nll_loss_grad_2D(dx_data, dout_data, label_data, weight_data,
                       total_weight_data, batch_size, n_classes, in_dim2,
                       in_dim3, reduction, ignore_index);
    }
  }
};

}  // namespace operators
}  // namespace paddle
