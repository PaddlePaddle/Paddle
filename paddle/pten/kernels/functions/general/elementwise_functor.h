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
#include "paddle/pten/kernels/functions/blas/elementwise.h"
#include "paddle/pten/kernels/functions/eigen/elementwise.h"

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
struct SameDimsSubFunctor {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z);
};

template <typename DevCtx, typename T>
struct SameDimsSubFunctor<
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
struct SameDimsSubFunctor<
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
struct SubFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const { return a - b; }
};
template <typename T>
struct InverseSubFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const { return b - a; }
};

}  // namespace general
}  // namespace pten
