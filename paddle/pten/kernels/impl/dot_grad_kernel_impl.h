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

#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/kernels/hybird/eigen/common.h"

#include "paddle/pten/kernels/complex_kernel.h"

#include "paddle/fluid/operators/eigen/eigen_function.h"
#include "paddle/fluid/operators/math/complex_functors.h"

namespace pten {

template <typename DeviceContext, typename T, typename Enabel = void>
struct DotGradFunction {
  void operator()(const DeviceContext& ctx,
                  const DenseTensor* tensor_x,
                  const DenseTensor* tensor_y,
                  const DenseTensor* tensor_dout,
                  DenseTensor* tensor_dx,
                  DenseTensor* tensor_dy);
};

template <typename DeviceContext, typename T>
struct DotGradFunction<DeviceContext,
                       T,
                       paddle::operators::math::EnableComplex<T>> {
  void operator()(const DeviceContext& ctx,
                  const DenseTensor* tensor_x,
                  const DenseTensor* tensor_y,
                  const DenseTensor* tensor_dout,
                  DenseTensor* tensor_dx,
                  DenseTensor* tensor_dy) {
#if defined(__NVCC__) || defined(__HIPCC__)
    if (1 == tensor_dout->dims().size()) {
      auto dout = EigenVector<T>::Flatten(*tensor_dout);

      if (tensor_dx) {
        auto y = EigenVector<T>::Flatten(*tensor_y);
        auto& dev = *ctx.eigen_device();
        Eigen::DSizes<int, 1> size(tensor_dx->numel());

        pten::Conj<T, DeviceContext>(ctx, *tensor_y, tensor_dx);

        auto dx = EigenVector<T>::Flatten(*tensor_dx);
        dx.device(dev) = dx * dout.broadcast(size);
      }

      if (tensor_dy) {
        auto x = EigenVector<T>::Flatten(*tensor_x);
        auto& dev = *ctx.eigen_device();
        Eigen::DSizes<int, 1> size(tensor_dy->numel());

        pten::Conj<T, DeviceContext>(ctx, *tensor_x, tensor_dy);

        auto dy = EigenVector<T>::Flatten(*tensor_dy);
        dy.device(dev) = dy * dout.broadcast(size);
      }
    } else {
      auto dout = EigenMatrix<T>::From(*tensor_dout);

      if (tensor_dx) {
        tensor_dx->mutable_data<T>();
        auto y = EigenMatrix<T>::From(*tensor_y);
        auto& dev = *ctx.eigen_device();
        Eigen::DSizes<int, 2> size(1, tensor_dx->dims()[1]);

        pten::Conj<T, DeviceContext>(ctx, *tensor_y, tensor_dx);

        auto dx = EigenMatrix<T>::From(*tensor_dx);
        dx.device(dev) = dx * dout.broadcast(size);
      }

      if (tensor_dy) {
        tensor_dy->mutable_data<T>();
        auto x = EigenMatrix<T>::From(*tensor_x);
        auto& dev = *ctx.eigen_device();
        Eigen::DSizes<int, 2> size(1, tensor_dy->dims()[1]);

        pten::Conj<T, DeviceContext>(ctx, *tensor_x, tensor_dy);

        auto dy = EigenMatrix<T>::From(*tensor_dy);
        dy.device(dev) = dy * dout.broadcast(size);
      }
    }
#else
    const auto* data_dout = tensor_dout->data<T>();

    if (tensor_dx) {
      auto* data_dx = tensor_dx->mutable_data<T>();
      const auto* data_y = tensor_y->data<T>();
      const DDim& dim = tensor_x->dims();
      size_t N = static_cast<size_t>(paddle::framework::product(dim));

      auto step = dim[dim.size() - 1];

      int s = -1;
      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_dx[i] = T(data_y[i].real, -data_y[i].imag) * data_dout[s];
      }
    }

    if (tensor_dy) {
      auto* data_dy = tensor_dy->mutable_data<T>();
      const auto* data_x = tensor_x->data<T>();
      const DDim& dim = tensor_y->dims();
      size_t N = static_cast<size_t>(paddle::framework::product(dim));

      auto step = dim[dim.size() - 1];

      int s = -1;
      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_dy[i] = T(data_x[i].real, -data_x[i].imag) * data_dout[s];
      }
    }
#endif
  }
};

template <typename DeviceContext, typename T>
struct DotGradFunction<DeviceContext,
                       T,
                       paddle::operators::math::DisableComplex<T>> {
  void operator()(const DeviceContext& ctx,
                  const DenseTensor* tensor_x,
                  const DenseTensor* tensor_y,
                  const DenseTensor* tensor_dout,
                  DenseTensor* tensor_dx,
                  DenseTensor* tensor_dy) {
#if defined(__NVCC__) || defined(__HIPCC__)
    if (1 == tensor_dout->dims().size()) {
      auto dout = EigenVector<T>::Flatten(*tensor_dout);
      if (tensor_dx) {
        auto y = pten::EigenVector<T>::Flatten(*tensor_y);
        auto dx = pten::EigenVector<T>::Flatten(*tensor_dx);
        auto& dev = *ctx.eigen_device();
        Eigen::DSizes<int, 1> size(tensor_dx->numel());
        dx.device(dev) = y * dout.broadcast(size);
      }

      if (tensor_dy) {
        auto x = pten::EigenVector<T>::Flatten(*tensor_x);
        auto dy = pten::EigenVector<T>::Flatten(*tensor_dy);
        auto& dev = *ctx.eigen_device();
        Eigen::DSizes<int, 1> size(tensor_dy->numel());
        dy.device(dev) = x * dout.broadcast(size);
      }
    } else {
      auto dout = EigenMatrix<T>::From(*tensor_dout);

      if (tensor_dx) {
        tensor_dx->mutable_data<T>();
        auto y = pten::EigenMatrix<T>::From(*tensor_y);
        auto dx = pten::EigenMatrix<T>::From(*tensor_dx);
        auto& dev = *ctx.eigen_device();
        Eigen::DSizes<int, 2> size(1, tensor_dx->dims()[1]);
        dx.device(dev) = y * dout.broadcast(size);
      }

      if (tensor_dy) {
        tensor_dy->mutable_data<T>();
        auto x = pten::EigenMatrix<T>::From(*tensor_x);
        auto dy = pten::EigenMatrix<T>::From(*tensor_dy);
        auto& dev = *ctx.eigen_device();
        Eigen::DSizes<int, 2> size(1, tensor_dy->dims()[1]);
        dy.device(dev) = x * dout.broadcast(size);
      }
    }
#else
    auto const *x = tensor_x->data<T>(), *y = tensor_y->data<T>(),
               *dz = tensor_dout->data<T>();
    auto&& d = tensor_x->dims();
    auto const N = tensor_x->numel();
    auto const B = d[d.size() - 1];

    if (tensor_dx) {
      auto* dx = tensor_dx->mutable_data<T>();
      for (auto j = 0; j < N / B; ++j) {
        auto const ss = dz[j];
        for (auto i = 0; i < B; ++i) *dx++ = *y++ * ss;
      }
    }

    if (tensor_dy) {
      auto* dy = tensor_dy->mutable_data<T>();
      for (auto j = 0; j < N / B; ++j) {
        auto const ss = dz[j];
        for (auto i = 0; i < B; i++) *dy++ = *x++ * ss;
      }
    }
#endif
  }
};

template <typename T, typename ContextT>
void DotGrad(const ContextT& dev_ctx,
             const DenseTensor& x,
             const DenseTensor& y,
             const DenseTensor& dout,
             DenseTensor* dx,
             DenseTensor* dy) {
  if (dx) {
    dx->mutable_data<T>();
  }
  if (dy) {
    dy->mutable_data<T>();
  }
  DotGradFunction<ContextT, T>()(dev_ctx, &x, &y, &dout, dx, dy);
}

}  // namespace pten
