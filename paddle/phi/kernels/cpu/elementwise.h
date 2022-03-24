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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

// FORWARD CODE

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
    auto blas = phi::funcs::GetBlas<DevCtx, T>(dev_ctx);
    blas.VADD(
        x.numel(), x.data<T>(), y.data<T>(), dev_ctx.template Alloc<T>(z));
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
    dev_ctx.template Alloc<T>(z);
    auto eigen_x = phi::EigenVector<T>::Flatten(x);
    auto eigen_y = phi::EigenVector<T>::Flatten(y);
    auto eigen_z = phi::EigenVector<T>::Flatten(*z);
    auto& place = *dev_ctx.eigen_device();
    eigen_z.device(place) = eigen_x + eigen_y;
  }
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
    auto blas = phi::funcs::GetBlas<DevCtx, T>(dev_ctx);
    blas.VSUB(
        x.numel(), x.data<T>(), y.data<T>(), dev_ctx.template Alloc<T>(z));
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
    auto eigen_x = phi::EigenVector<T>::Flatten(x);
    auto eigen_y = phi::EigenVector<T>::Flatten(y);
    auto eigen_z = phi::EigenVector<T>::Flatten(*z);
    auto& place = *dev_ctx.eigen_device();
    eigen_z.device(place) = eigen_x - eigen_y;
  }
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
    phi::errors::InvalidArgument(
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
    auto blas = phi::funcs::GetBlas<DevCtx, T>(dev_ctx);
    blas.VDIV(
        x.numel(), x.data<T>(), y.data<T>(), dev_ctx.template Alloc<T>(z));
  }
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
    auto blas = phi::funcs::GetBlas<DevCtx, T>(dev_ctx);
    blas.VMUL(
        x.numel(), x.data<T>(), y.data<T>(), dev_ctx.template Alloc<T>(z));
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
    auto eigen_x = phi::EigenVector<T>::Flatten(x);
    auto eigen_y = phi::EigenVector<T>::Flatten(y);
    auto eigen_z = phi::EigenVector<T>::Flatten(*z);
    auto& place = *dev_ctx.eigen_device();
    eigen_z.device(place) = eigen_x * eigen_y;
  }
};

template <typename Functor>
struct SameDimsElementwiseCompute {
  void operator()(const CPUContext& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z) {
    Functor()(dev_ctx, x, y, z);
  }
};

}  // namespace phi
