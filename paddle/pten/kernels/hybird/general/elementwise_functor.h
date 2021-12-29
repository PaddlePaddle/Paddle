/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/hostdevice.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/kernels/hybird/blas/elementwise.h"
#include "paddle/pten/kernels/hybird/eigen/elementwise.h"

namespace pten {
namespace general {

// Define the binary functors used in elementwise ops.

// Add
template <typename DevCtx, typename T, class Enable = void>
struct SameDimsAddFunctor {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z);
};

template <typename DevCtx, typename T>
struct SameDimsAddFunctor<
    DevCtx,
    T,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z) {
    blas::ElementwiseAdd<DevCtx, T>(dev_ctx, x, y, z);
  }
};

template <typename DevCtx, typename T>
struct SameDimsAddFunctor<
    DevCtx,
    T,
    typename std::enable_if<!std::is_floating_point<T>::value>::type> {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z) {
    eigen::ElementwiseAdd<DevCtx, T>(dev_ctx, x, y, z);
  }
};

template <typename T>
struct AddFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const { return a + b; }
};
template <typename T>
struct InverseAddFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const { return b + a; }
};

// Subtract
template <typename DevCtx, typename T, class Enable = void>
struct SameDimsSubtractFunctor {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z);
};

template <typename DevCtx, typename T>
struct SameDimsSubtractFunctor<
    DevCtx,
    T,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z) {
    blas::ElementwiseSub<DevCtx, T>(dev_ctx, x, y, z);
  }
};

template <typename DevCtx, typename T>
struct SameDimsSubtractFunctor<
    DevCtx,
    T,
    typename std::enable_if<!std::is_floating_point<T>::value>::type> {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z) {
    eigen::ElementwiseSub<DevCtx, T>(dev_ctx, x, y, z);
  }
};

template <typename T>
struct SubtractFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const { return a - b; }
};
template <typename T>
struct InverseSubtractFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const { return b - a; }
};

// Divide
template <typename DevCtx, typename T, class Enable = void>
struct SameDimsDivideFunctor {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z);
};

template <typename DevCtx, typename T>
struct SameDimsDivideFunctor<
    DevCtx,
    T,
    typename std::enable_if<!std::is_floating_point<T>::value>::type> {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z) {
    paddle::platform::errors::InvalidArgument(
        "If use SameDimsDivideFunctor, template args(T) must be floating "
        "point. ");
  }
};

template <typename DevCtx, typename T>
struct SameDimsDivideFunctor<
    DevCtx,
    T,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z) {
    blas::ElementwiseDiv<DevCtx, T>(dev_ctx, x, y, z);
  }
};

#define DIV_ERROR_INFO                                             \
  "InvalidArgumentError: Integer division by zero encountered in " \
  "(floor) divide. Please check the input value."

template <typename T, typename Enable = void>
struct DivideFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const { return a / b; }
};

template <typename T>
struct DivideFunctor<
    T,
    typename std::enable_if<std::is_integral<T>::value>::type> {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const {
    // For int32/int64, need to check whether the divison is zero.
    PADDLE_ENFORCE(b != 0, DIV_ERROR_INFO);
    return a / b;
  }
};

template <typename T, typename Enable = void>
struct InverseDivideFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const { return b / a; }
};

// Multiply
template <typename DevCtx, typename T, class Enable = void>
struct SameDimsMultiplyFunctor {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z);
};

template <typename DevCtx, typename T>
struct SameDimsMultiplyFunctor<
    DevCtx,
    T,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z) {
    blas::ElementwiseMul<DevCtx, T>(dev_ctx, x, y, z);
  }
};

template <typename DevCtx, typename T>
struct SameDimsMultiplyFunctor<
    DevCtx,
    T,
    typename std::enable_if<!std::is_floating_point<T>::value>::type> {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z) {
    eigen::ElementwiseMul<DevCtx, T>(dev_ctx, x, y, z);
  }
};
template <typename T>
struct MultiplyFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const { return a * b; }
};
template <typename T>
struct InverseMultiplyFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const { return b * a; }
};

}  // namespace general
}  // namespace pten
