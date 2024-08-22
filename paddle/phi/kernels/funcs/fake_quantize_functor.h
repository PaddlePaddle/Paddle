/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/common/hostdevice.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/transform.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/impl/clip_kernel_impl.h"

namespace phi {
namespace funcs {

template <typename T>
inline HOSTDEVICE T inverse(T s) {
  T eps = static_cast<T>(1e-6);
  T one = static_cast<T>(1.0);
  return s <= static_cast<T>(1e-30) ? one / (s + eps) : one / s;
}

template <typename T>
struct Compare {
  bool operator()(const T a, const T b) { return (std::abs(a) < std::abs(b)); }
};

template <typename T>
inline HOSTDEVICE T roundWithTiesToEven(T x) {
  T xLower = floor(x);
  T xUpper = ceil(x);
  // x is in interval [xl,xu]. Choose closest of two bounds, breaking ties to
  // even.
  T dLower = x - xLower;
  T dUpper = xUpper - x;
  return static_cast<T>(
      (dLower == dUpper ? fmod(xLower, 2.0F) == 0.0F : dLower < dUpper)
          ? xLower
          : xUpper);
}

template <typename T>
class QuantTensorFunctor {
 public:
  explicit QuantTensorFunctor(const T qmax, const T inv_s)
      : qmax_(qmax), inv_s_(inv_s) {}
  HOSTDEVICE T operator()(const T x) const {
    T out = qmax_ * inv_s_ * x;
    if (qmax_ == static_cast<T>(448)) {
      out = float8_e4m3fn(out);
    } else if (qmax_ == static_cast<T>(57344)) {
      out = float8_e5m2(out);
    } else {
      out = roundWithTiesToEven(out);
    }
    T max_bound = qmax_;
    T min_bound = -qmax_ - static_cast<T>(1);
    if (qmax_ == static_cast<T>(448) || qmax_ == static_cast<T>(57344)) {
      min_bound = -qmax_;
    }
    out = out > max_bound ? max_bound : out;
    out = out < min_bound ? min_bound : out;
    return out;
  }

 private:
  T qmax_;
  T inv_s_;
};

template <typename Context, typename T>
class FindAbsMaxFunctor {
 public:
  void operator()(const Context &ctx, const T *in, const int num, T *out);
};

template <typename Context, typename T>
class ClipAndFakeQuantFunctor {
 public:
  void operator()(const Context &ctx,
                  const DenseTensor &in,
                  const DenseTensor &scale,
                  const int qmax,
                  const int round_type,
                  DenseTensor *out);
};

template <typename Context, typename T>
class FindMovingAverageAbsMaxFunctor {
 public:
  void operator()(const Context &ctx,
                  const DenseTensor &in_accum,
                  const DenseTensor &in_state,
                  const T *cur_scale,
                  const float rate,
                  DenseTensor *out_state,
                  DenseTensor *out_accum,
                  DenseTensor *out_scale);
};

template <typename Context, typename T>
class FindChannelAbsMaxFunctor {
 public:
  void operator()(const Context &ctx,
                  const DenseTensor &in_tensor,
                  const int quant_axis,
                  T *out_abs_max);
};

template <typename Context, typename T>
class ChannelClipAndFakeQuantFunctor {
 public:
  void operator()(const Context &ctx,
                  const DenseTensor &in,
                  const DenseTensor &scale,
                  const int qmax,
                  const int round_type,
                  const int quant_axis,
                  DenseTensor *out);
};

template <typename Context, typename T>
class ChannelClipFakeQuantDequantFunctor {
 public:
  void operator()(const Context &ctx,
                  const DenseTensor &in,
                  const DenseTensor &scale,
                  const int bin_cnt,
                  const int round_type,
                  const int quant_axis,
                  DenseTensor *out);
};

template <typename Context, typename T>
class FindRangeAbsMaxFunctor {
 public:
  void operator()(const Context &ctx,
                  const DenseTensor &cur_scale,
                  const DenseTensor &last_scale,
                  const DenseTensor &iter,
                  const int window_size,
                  DenseTensor *scales_arr,
                  DenseTensor *out_scale);
};

template <typename Context, typename T>
class ClipAndFakeQuantDequantFunctor {
 public:
  void operator()(const Context &ctx,
                  const DenseTensor &in,
                  const DenseTensor &scale,
                  const int bin_cnt,
                  int round_type,
                  DenseTensor *out);
};

}  // namespace funcs
}  // namespace phi
