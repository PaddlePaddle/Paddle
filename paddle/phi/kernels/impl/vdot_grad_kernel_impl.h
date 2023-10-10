// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace phi {
template <typename DeviceContext, typename T, typename Enabel = void>
struct VdotGradFunction {
  void operator()(const DeviceContext& ctx,
                  const DenseTensor* tensor_x,
                  const DenseTensor* tensor_y,
                  const DenseTensor* tensor_dout,
                  DenseTensor* tensor_dx,
                  DenseTensor* tensor_dy);
};

template <typename DeviceContext, typename T>
struct VdotGradFunction<DeviceContext, T, phi::funcs::EnableComplex<T>> {
  void operator()(const DeviceContext& ctx,
                  const DenseTensor* tensor_x,
                  const DenseTensor* tensor_y,
                  const DenseTensor* tensor_dout,
                  DenseTensor* tensor_dx,
                  DenseTensor* tensor_dy) {
    if (1 >= tensor_dout->dims().size()) {
      if (tensor_dx) {
        ctx.template Alloc<T>(tensor_dx);
        auto y = EigenVector<T>::Flatten(*tensor_y);
        auto& dev = *ctx.eigen_device();
        Eigen::DSizes<int, 1> size(tensor_dx->numel());

        auto dout_conj = EigenVector<T>::Flatten(*tensor_dout).conjugate();
        auto dx = EigenVector<T>::Flatten(*tensor_dx);
        dx.device(dev) = y * dout_conj.broadcast(size);
      }

      if (tensor_dy) {
        ctx.template Alloc<T>(tensor_dy);
        auto x = EigenVector<T>::Flatten(*tensor_x);
        auto& dev = *ctx.eigen_device();
        Eigen::DSizes<int, 1> size(tensor_dy->numel());

        auto dout = EigenVector<T>::Flatten(*tensor_dout);
        auto dy = EigenVector<T>::Flatten(*tensor_dy);
        dy.device(dev) = x * dout.broadcast(size);
      }
    } else {
      if (tensor_dx) {
        ctx.template Alloc<T>(tensor_dx);
        auto y = EigenMatrix<T>::From(*tensor_y);
        auto& dev = *ctx.eigen_device();
        Eigen::DSizes<int, 2> size(1, tensor_dx->dims()[1]);

        auto dout_conj = EigenMatrix<T>::From(*tensor_dout).conjugate();
        auto dx = EigenMatrix<T>::From(*tensor_dx);
        dx.device(dev) = y * dout_conj.broadcast(size);
      }

      if (tensor_dy) {
        ctx.template Alloc<T>(tensor_dy);
        auto x = EigenMatrix<T>::From(*tensor_x);
        auto& dev = *ctx.eigen_device();
        Eigen::DSizes<int, 2> size(1, tensor_dy->dims()[1]);

        auto dout = EigenMatrix<T>::From(*tensor_dout);
        auto dy = EigenMatrix<T>::From(*tensor_dy);
        dy.device(dev) = x * dout.broadcast(size);
      }
    }
  }
};

template <typename DeviceContext, typename T>
struct VdotGradFunction<DeviceContext, T, phi::funcs::DisableComplex<T>> {
  void operator()(const DeviceContext& ctx,
                  const DenseTensor* tensor_x,
                  const DenseTensor* tensor_y,
                  const DenseTensor* tensor_dout,
                  DenseTensor* tensor_dx,
                  DenseTensor* tensor_dy) {
    if (1 >= tensor_dout->dims().size()) {
      auto dout = EigenVector<T>::Flatten(*tensor_dout);
      if (tensor_dx) {
        ctx.template Alloc<T>(tensor_dx);
        auto y = EigenVector<T>::Flatten(*tensor_y);
        auto dx = EigenVector<T>::Flatten(*tensor_dx);
        auto& dev = *ctx.eigen_device();
        Eigen::DSizes<int, 1> size(tensor_dx->numel());
        dx.device(dev) = y * dout.broadcast(size);
      }

      if (tensor_dy) {
        ctx.template Alloc<T>(tensor_dy);
        auto x = EigenVector<T>::Flatten(*tensor_x);
        auto dy = EigenVector<T>::Flatten(*tensor_dy);
        auto& dev = *ctx.eigen_device();
        Eigen::DSizes<int, 1> size(tensor_dy->numel());
        dy.device(dev) = x * dout.broadcast(size);
      }
    } else {
      auto dout = EigenMatrix<T>::From(*tensor_dout);

      if (tensor_dx) {
        ctx.template Alloc<T>(tensor_dx);
        auto y = EigenMatrix<T>::From(*tensor_y);
        auto dx = EigenMatrix<T>::From(*tensor_dx);
        auto& dev = *ctx.eigen_device();
        Eigen::DSizes<int, 2> size(1, tensor_dx->dims()[1]);
        dx.device(dev) = y * dout.broadcast(size);
      }

      if (tensor_dy) {
        ctx.template Alloc<T>(tensor_dy);
        auto x = EigenMatrix<T>::From(*tensor_x);
        auto dy = EigenMatrix<T>::From(*tensor_dy);
        auto& dev = *ctx.eigen_device();
        Eigen::DSizes<int, 2> size(1, tensor_dy->dims()[1]);
        dy.device(dev) = x * dout.broadcast(size);
      }
    }
  }
};

template <typename T, typename Context>
void VdotGradKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    const DenseTensor& dout,
                    DenseTensor* dx,
                    DenseTensor* dy) {
  VdotGradFunction<Context, T>()(dev_ctx, &x, &y, &dout, dx, dy);
}

}  // namespace phi
