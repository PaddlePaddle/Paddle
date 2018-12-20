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
#include <limits>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/hostdevice.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
struct TolerableValue {
  HOSTDEVICE T operator()(const T& x) const {
    PADDLE_ASSERT(std::is_floating_point<T>::value);
    const T kApproInf = 1e20;

    if (x == INFINITY) return kApproInf;
    if (x == -INFINITY) return -kApproInf;
    return x;
  }
};

// NOTE(dzh): float16 value clip behave different.
// 1. Our ValueClipping has a  hardcore threshold 1e20
// for float number. 1e20 will resulting in overflow in float16.
// 2. float16 should expose the the real number overflow to python.
// because mixed-training depends the inf/nan value to determine
// if the scale value will be adjusted.
// Also. In standard implementation of cross entropy, other
// framework not has the ValueClipping.
template <>
struct TolerableValue<platform::float16> {
  HOSTDEVICE platform::float16 operator()(const platform::float16& x) const {
    if (platform::isfinite(x))
      return x;
    else if (x > static_cast<platform::float16>(0))
      return std::numeric_limits<platform::float16>::max();
    else
      return std::numeric_limits<platform::float16>::min();
  }
};

/* UNDERSTAND: OK, finally, functor is but a function, universally, but if it
has Tensors as its parameter, it must have a DeviceContext as its parameter
too, but this does not make it Restricted to Tensor functions */
template <typename DeviceContext, typename T>
class CrossEntropyMultiLabelFunctor {
 public:
  void operator()(const DeviceContext& context, framework::Tensor* out,
                  const framework::Tensor* prob,
                  const framework::Tensor* labels, const int ignore_index);
};

/*
template <typename T>
class XeMlGradFunctor{
public:
  XeMlGradFunctor(T* dx,
                  const T* dy,          // NOLINT
                  const T* x,           // NOLINT
                  const int64_t* label, // NOLINT
                  size_t num_classes, size_t num_true, size_t ignore_index)
    : dx_(dx),
      dy_(dy),
      x_(x),
      label_(label),
      num_classes_(num_classes),
      num_true_(num_true),
      ignore_index_(ignore_index) {}

  HOSTDEVICE void operator()(size_t sample_id) {
    // UNDERSTAND: it only computes for a single example, batch it by loop
    for (size_t x_offset = sample_id * num_classes_;
         x_offset < (sample_id + 1) * num_classes_; ++x_offset) {
      dx_[x_offset] = static_cast<T>(0);
    }
    for (size_t j = 0; j < num_true_; ++j) {
      auto lbl_offset = sample_id * num_true_ + j;
      auto x_is_true_offset = sample_id * num_classes_ + label_[lbl_offset];
      dx_[x_is_true_offset] -= dy_[sample_id] / x_[x_is_true_offset];
    }
  }

private:
  T* dx_;
  const T* dy_;
  const T* x_;
  const int64_t* label_;
  std::size_t num_classes_;
  std::size_t num_true_;
  std::size_t ignore_index_;
};
*/
}  // namespace math
}  // namespace operators
}  // namespace paddle
