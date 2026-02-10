// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/atan2_grad_kernel.h"
#include "paddle/phi/kernels/broadcast_tensors_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {

// dx1 = dout * x2 / ((x1)^2 + (x2)^2)
// dx2 = - dout * x1 / ((x1)^2 + (x2)^2)
template <typename T>
struct Atan2GradFunctor {
  Atan2GradFunctor(
      const T* x1, const T* x2, const T* dout, T* dx1, T* dx2, int64_t numel)
      : x1_(x1), x2_(x2), dout_(dout), dx1_(dx1), dx2_(dx2), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    float x1 = static_cast<float>(x1_[idx]);
    float x2 = static_cast<float>(x2_[idx]);
    float x = x1 * x1 + x2 * x2;
    if (dx1_) {
      dx1_[idx] = static_cast<T>(static_cast<float>(dout_[idx]) * x2 / x);
    }
    if (dx2_) {
      dx2_[idx] = static_cast<T>(-static_cast<float>(dout_[idx]) * x1 / x);
    }
  }

  const T* x1_;
  const T* x2_;
  const T* dout_;
  T* dx1_;
  T* dx2_;
  int64_t numel_;
};

template <>
struct Atan2GradFunctor<double> {
  Atan2GradFunctor(const double* x1,
                   const double* x2,
                   const double* dout,
                   double* dx1,
                   double* dx2,
                   int64_t numel)
      : x1_(x1), x2_(x2), dout_(dout), dx1_(dx1), dx2_(dx2), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    auto x = x1_[idx] * x1_[idx] + x2_[idx] * x2_[idx];
    if (dx1_) {
      dx1_[idx] = dout_[idx] * x2_[idx] / x;
    }
    if (dx2_) {
      dx2_[idx] = -dout_[idx] * x1_[idx] / x;
    }
  }

  const double* x1_;
  const double* x2_;
  const double* dout_;
  double* dx1_;
  double* dx2_;
  int64_t numel_;
};

template <typename T, typename Context>
void Atan2GradKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     const DenseTensor& out_grad,
                     DenseTensor* x_grad,
                     DenseTensor* y_grad) {
  if (out_grad.numel() == 0) {
    if (x_grad) {
      dev_ctx.template Alloc<T>(x_grad);
      if (x_grad->numel() != 0) {
        Full<T, Context>(dev_ctx, x_grad->dims(), 0, x_grad);
      }
    }
    if (y_grad) {
      dev_ctx.template Alloc<T>(y_grad);
      if (y_grad->numel() != 0) {
        Full<T, Context>(dev_ctx, y_grad->dims(), 0, y_grad);
      }
    }
    return;
  }

  if (x.dims() == y.dims() && x.dims() == out_grad.dims()) {
    auto numel = x.numel();
    auto x_data = x.data<T>();
    auto y_data = y.data<T>();
    auto out_grad_data = out_grad.data<T>();

    auto* x_grad_data = x_grad ? dev_ctx.template Alloc<T>(
                                     x_grad, size_t(x.numel() * sizeof(T)))
                               : nullptr;
    auto* y_grad_data = y_grad ? dev_ctx.template Alloc<T>(
                                     y_grad, size_t(y.numel() * sizeof(T)))
                               : nullptr;

    phi::funcs::ForRange<Context> for_range(dev_ctx, numel);
    phi::Atan2GradFunctor<T> functor(
        x_data, y_data, out_grad_data, x_grad_data, y_grad_data, numel);
    for_range(functor);
  } else {
    DenseTensor b_x, b_y;
    b_x.Resize(out_grad.dims());
    b_y.Resize(out_grad.dims());

    std::vector<const DenseTensor*> inputs = {&x, &y};
    std::vector<DenseTensor*> outputs = {&b_x, &b_y};
    phi::BroadcastTensorsKernel<T, Context>(dev_ctx, inputs, outputs);

    DenseTensor dx_b, dy_b;
    T* dx_b_data = nullptr;
    T* dy_b_data = nullptr;
    std::vector<int64_t> x_axes, y_axes;

    if (x_grad) {
      int in_rank = x.dims().size();
      int out_rank = out_grad.dims().size();
      int diff = out_rank - in_rank;
      for (int i = 0; i < diff; ++i) x_axes.push_back(i);
      for (int i = 0; i < in_rank; ++i) {
        if (x.dims()[i] == 1 && out_grad.dims()[i + diff] > 1) {
          x_axes.push_back(i + diff);
        }
      }
      if (x_axes.empty()) {
        dev_ctx.template Alloc<T>(x_grad);
        dx_b_data = x_grad->data<T>();
      } else {
        dx_b.Resize(out_grad.dims());
        dx_b_data = dev_ctx.template Alloc<T>(&dx_b);
      }
    }

    if (y_grad) {
      int in_rank = y.dims().size();
      int out_rank = out_grad.dims().size();
      int diff = out_rank - in_rank;
      for (int i = 0; i < diff; ++i) y_axes.push_back(i);
      for (int i = 0; i < in_rank; ++i) {
        if (y.dims()[i] == 1 && out_grad.dims()[i + diff] > 1) {
          y_axes.push_back(i + diff);
        }
      }
      if (y_axes.empty()) {
        dev_ctx.template Alloc<T>(y_grad);
        dy_b_data = y_grad->data<T>();
      } else {
        dy_b.Resize(out_grad.dims());
        dy_b_data = dev_ctx.template Alloc<T>(&dy_b);
      }
    }

    auto numel = out_grad.numel();
    phi::funcs::ForRange<Context> for_range(dev_ctx, numel);
    phi::Atan2GradFunctor<T> functor(b_x.data<T>(),
                                     b_y.data<T>(),
                                     out_grad.data<T>(),
                                     dx_b_data,
                                     dy_b_data,
                                     numel);
    for_range(functor);

    if (x_grad && !x_axes.empty()) {
      phi::SumKernel<T, Context>(
          dev_ctx, dx_b, phi::IntArray(x_axes), x_grad->dtype(), false, x_grad);
      x_grad->Resize(x.dims());
    }

    if (y_grad && !y_axes.empty()) {
      phi::SumKernel<T, Context>(
          dev_ctx, dy_b, phi::IntArray(y_axes), y_grad->dtype(), false, y_grad);
      y_grad->Resize(y.dims());
    }
  }
}

}  // namespace phi
