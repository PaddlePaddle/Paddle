/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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
#ifdef PADDLE_WITH_XPU
#include <algorithm>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "xpu/refactor/math.h"

namespace phi {

template <typename T, typename XPUType>
void XPUElementwise(const XPUContext& dev_ctx,
                    const T* x_data,
                    const DDim& x_dims,
                    const T* y_data,
                    const DDim& y_dims,
                    int axis,
                    T* z_data,
                    std::function<int(xpu::Context*,
                                      const XPUType*,
                                      const XPUType*,
                                      XPUType*,
                                      const std::vector<int>&,
                                      const std::vector<int>&)> func) {
  int max_dim = std::max(x_dims.size(), y_dims.size());
  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);

  PADDLE_ENFORCE_GE(
      axis,
      0,
      errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LE(
      axis,
      max_dim,
      errors::InvalidArgument(
          "Axis should be less than or equal to %d, but received axis is %d.",
          max_dim,
          axis));
  std::vector<int> x_dims_vec(max_dim, 1);
  std::vector<int> y_dims_vec(max_dim, 1);
  if (x_dims.size() == max_dim) {
    for (int i = 0; i < max_dim; i++) {
      x_dims_vec[i] = x_dims[i];
    }
  } else {
    for (int i = 0; i < x_dims.size(); i++) {
      x_dims_vec[i + axis] = x_dims[i];
    }
  }
  if (y_dims.size() == max_dim) {
    for (int i = 0; i < max_dim; i++) {
      y_dims_vec[i] = y_dims[i];
    }
  } else {
    for (int i = 0; i < y_dims.size(); i++) {
      y_dims_vec[i + axis] = y_dims[i];
    }
  }

  int ret = xpu::SUCCESS;

  // For [2, 3] + [] --> [2, 3] + [1, 1]
  // For [] + [2, 3] --> [1, 1] + [2, 3]
  // For [] + [], Use [1] + [1] to replace [], because xpu not support []
  if (x_dims_vec.size() == 0) {
    x_dims_vec = std::vector<int>({1});
  }

  if (y_dims_vec.size() == 0) {
    y_dims_vec = std::vector<int>({1});
  }

  ret = func(dev_ctx.x_context(),
             reinterpret_cast<const XPUType*>(x_data),
             reinterpret_cast<const XPUType*>(y_data),
             reinterpret_cast<XPUType*>(z_data),
             x_dims_vec,
             y_dims_vec);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "elementwise");
}

template <typename T, typename XPUType>
void XPUElementwise(const XPUContext& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    int axis,
                    DenseTensor* z,
                    std::function<int(xpu::Context*,
                                      const XPUType*,
                                      const XPUType*,
                                      XPUType*,
                                      const std::vector<int>&,
                                      const std::vector<int>&)> func) {
  dev_ctx.template Alloc<T>(z);
  const DDim& x_dims = x.dims();
  const DDim& y_dims = y.dims();

  const T* x_data = x.data<T>();
  const T* y_data = y.data<T>();
  T* z_data = z->data<T>();

  XPUElementwise<T, XPUType>(
      dev_ctx, x_data, x_dims, y_data, y_dims, axis, z_data, func);
}

template <typename T, typename XPUType>
void XPUElementwiseGrad(const XPUContext& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        const DenseTensor& dz,
                        int axis,
                        DenseTensor* dx,
                        DenseTensor* dy,
                        std::function<int(xpu::Context*,
                                          const XPUType*,
                                          const XPUType*,
                                          const XPUType*,
                                          const XPUType*,
                                          XPUType*,
                                          XPUType*,
                                          const std::vector<int>&,
                                          const std::vector<int>&)> func,
                        bool use_x_y_data) {
  auto* z = &dz;
  const DDim& x_dims = x.dims();
  const DDim& y_dims = y.dims();
  int max_dim = std::max(x_dims.size(), y_dims.size());
  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  PADDLE_ENFORCE_GE(
      axis,
      0,
      errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LE(
      axis,
      max_dim,
      errors::InvalidArgument(
          "Axis should be less than or equal to %d, but received axis is %d.",
          max_dim,
          axis));
  std::vector<int> x_dims_vec(max_dim, 1);
  std::vector<int> y_dims_vec(max_dim, 1);
  if (x_dims.size() == max_dim) {
    for (int i = 0; i < max_dim; i++) {
      x_dims_vec[i] = x_dims[i];
    }
  } else {
    for (int i = 0; i < x_dims.size(); i++) {
      x_dims_vec[i + axis] = x_dims[i];
    }
  }
  if (y_dims.size() == max_dim) {
    for (int i = 0; i < max_dim; i++) {
      y_dims_vec[i] = y_dims[i];
    }
  } else {
    for (int i = 0; i < y_dims.size(); i++) {
      y_dims_vec[i + axis] = y_dims[i];
    }
  }

  const T* x_data = use_x_y_data ? x.data<T>() : z->data<T>();
  const T* y_data = use_x_y_data ? y.data<T>() : z->data<T>();
  const T* z_data = z->data<T>();

  const T* dz_data = dz.data<T>();
  T* dx_data = nullptr;
  T* dy_data = nullptr;

  if (dx) {
    dx_data = dev_ctx.template Alloc<T>(dx);
  }
  if (dy) {
    dy_data = dev_ctx.template Alloc<T>(dy);
  }

  // use [1] to replace [], because xpu not support []
  if (x_dims_vec.size() == 0) {
    x_dims_vec = std::vector<int>({1});
  }

  if (y_dims_vec.size() == 0) {
    y_dims_vec = std::vector<int>({1});
  }

  int ret = func(dev_ctx.x_context(),
                 reinterpret_cast<const XPUType*>(x_data),
                 reinterpret_cast<const XPUType*>(y_data),
                 reinterpret_cast<const XPUType*>(z_data),
                 reinterpret_cast<const XPUType*>(dz_data),
                 reinterpret_cast<XPUType*>(dy_data),
                 reinterpret_cast<XPUType*>(dx_data),
                 x_dims_vec,
                 y_dims_vec);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "elementwise");
}

}  // namespace phi
#endif
