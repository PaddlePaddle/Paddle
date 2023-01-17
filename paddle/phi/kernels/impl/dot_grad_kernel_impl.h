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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace phi {

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
struct DotGradFunction<DeviceContext, T, phi::funcs::EnableComplex<T>> {
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

        ConjKernel<T, DeviceContext>(ctx, *tensor_y, tensor_dx);

        auto dx = EigenVector<T>::Flatten(*tensor_dx);
        dx.device(dev) = dx * dout.broadcast(size);
      }

      if (tensor_dy) {
        auto x = EigenVector<T>::Flatten(*tensor_x);
        auto& dev = *ctx.eigen_device();
        Eigen::DSizes<int, 1> size(tensor_dy->numel());

        ConjKernel<T, DeviceContext>(ctx, *tensor_x, tensor_dy);

        auto dy = EigenVector<T>::Flatten(*tensor_dy);
        dy.device(dev) = dy * dout.broadcast(size);
      }
    } else {
      auto dout = EigenMatrix<T>::From(*tensor_dout);

      if (tensor_dx) {
        ctx.template Alloc<T>(tensor_dx);
        auto y = EigenMatrix<T>::From(*tensor_y);
        auto& dev = *ctx.eigen_device();
        Eigen::DSizes<int, 2> size(1, tensor_dx->dims()[1]);

        ConjKernel<T, DeviceContext>(ctx, *tensor_y, tensor_dx);

        auto dx = EigenMatrix<T>::From(*tensor_dx);
        dx.device(dev) = dx * dout.broadcast(size);
      }

      if (tensor_dy) {
        ctx.template Alloc<T>(tensor_dy);
        auto x = EigenMatrix<T>::From(*tensor_x);
        auto& dev = *ctx.eigen_device();
        Eigen::DSizes<int, 2> size(1, tensor_dy->dims()[1]);

        ConjKernel<T, DeviceContext>(ctx, *tensor_x, tensor_dy);

        auto dy = EigenMatrix<T>::From(*tensor_dy);
        dy.device(dev) = dy * dout.broadcast(size);
      }
    }
#else
    const auto* data_dout = tensor_dout->data<T>();

    if (tensor_dx) {
      auto* data_dx = ctx.template Alloc<T>(tensor_dx);
      const auto* data_y = tensor_y->data<T>();
      const DDim& dim = tensor_x->dims();
      size_t N = static_cast<size_t>(phi::product(dim));

      auto step = dim[dim.size() - 1];

      int s = -1;
      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_dx[i] = T(data_y[i].real, -data_y[i].imag) * data_dout[s];
      }
    }

    if (tensor_dy) {
      auto* data_dy = ctx.template Alloc<T>(tensor_dy);
      const auto* data_x = tensor_x->data<T>();
      const DDim& dim = tensor_y->dims();
      size_t N = static_cast<size_t>(phi::product(dim));

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
struct DotGradFunction<DeviceContext, T, phi::funcs::DisableComplex<T>> {
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
        auto dx = EigenVector<T>::Flatten(*tensor_dx);
        auto& dev = *ctx.eigen_device();
        Eigen::DSizes<int, 1> size(tensor_dx->numel());
        dx.device(dev) = y * dout.broadcast(size);
      }

      if (tensor_dy) {
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
#else
    auto const *x = tensor_x->data<T>(), *y = tensor_y->data<T>(),
               *dz = tensor_dout->data<T>();
    auto&& d = tensor_x->dims();
    auto const N = tensor_x->numel();
    auto const B = d[d.size() - 1];

    if (tensor_dx) {
      auto* dx = ctx.template Alloc<T>(tensor_dx);
      for (auto j = 0; j < N / B; ++j) {
        auto const ss = dz[j];
        for (auto i = 0; i < B; ++i) *dx++ = *y++ * ss;
      }
    }

    if (tensor_dy) {
      auto* dy = ctx.template Alloc<T>(tensor_dy);
      for (auto j = 0; j < N / B; ++j) {
        auto const ss = dz[j];
        for (auto i = 0; i < B; i++) *dy++ = *x++ * ss;
      }
    }
#endif
  }
};

template <typename DeviceContext, typename T, typename Enabel = void>
struct DotDoubleGradFunction {
  void operator()(const DeviceContext& ctx,
                  const DenseTensor* tensor_x,
                  const DenseTensor* tensor_y,
                  const DenseTensor* tensor_dout,
                  const DenseTensor* tensor_ddx,
                  const DenseTensor* tensor_ddy,
                  DenseTensor* tensor_dx,
                  DenseTensor* tensor_dy,
                  DenseTensor* tensor_ddout);
};

template <typename DeviceContext, typename T>
struct DotDoubleGradFunction<DeviceContext, T, phi::funcs::EnableComplex<T>> {
  void operator()(const DeviceContext& ctx,
                  const DenseTensor* tensor_x,
                  const DenseTensor* tensor_y,
                  const DenseTensor* tensor_dout,
                  const DenseTensor* tensor_ddx,
                  const DenseTensor* tensor_ddy,
                  DenseTensor* tensor_dx,
                  DenseTensor* tensor_dy,
                  DenseTensor* tensor_ddout) {
#if defined(__NVCC__) || defined(__HIPCC__)
    if (1 == tensor_dout->dims().size()) {
      DenseTensor tensor_dout_help;
      auto& dev = *ctx.eigen_device();
      if (tensor_dx || tensor_dy) {
        tensor_dout_help = Conj<T, DeviceContext>(ctx, *tensor_dout);
      }
      if (tensor_dx) {
        auto ddy = EigenVector<T>::Flatten(*tensor_ddy);
        Eigen::DSizes<int, 1> size(tensor_ddy->numel());
        auto dx = EigenVector<T>::Flatten(*tensor_dx);
        auto dout = EigenVector<T>::Flatten(tensor_dout_help);
        dx.device(dev) = ddy * dout.broadcast(size);
      }

      if (tensor_dy) {
        auto ddx = EigenVector<T>::Flatten(*tensor_ddx);
        Eigen::DSizes<int, 1> size(tensor_ddx->numel());
        auto dy = EigenVector<T>::Flatten(*tensor_dy);
        auto dout = EigenVector<T>::Flatten(tensor_dout_help);
        dy.device(dev) = ddx * dout.broadcast(size);
      }

      if (tensor_ddout) {
        DenseTensor tensor_x_help = Conj<T, DeviceContext>(ctx, *tensor_x);
        DenseTensor tensor_y_help = Conj<T, DeviceContext>(ctx, *tensor_y);

        auto x = EigenVector<T>::Flatten(tensor_x_help);
        auto y = EigenVector<T>::Flatten(tensor_y_help);
        auto ddx = EigenVector<T>::Flatten(*tensor_ddx);
        auto ddy = EigenVector<T>::Flatten(*tensor_ddy);
        auto ddout = EigenVector<T>::Flatten(*tensor_ddout);
        ddout.device(dev) = (x * ddy + y * ddx).sum();
      }
    }
#else
    const auto* data_dout = tensor_dout->data<T>();

    if (tensor_dx) {
      auto* data_dx = ctx.template Alloc<T>(tensor_dx);
      const auto* data_ddy = tensor_ddy->data<T>();
      const DDim& dim = tensor_dx->dims();
      size_t N = static_cast<size_t>(product(dim));

      auto step = dim[dim.size() - 1];

      int s = -1;
      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_dx[i] = T(data_dout[s].real, -data_dout[s].imag) * data_ddy[i];
      }
    }

    if (tensor_dy) {
      auto* data_dy = ctx.template Alloc<T>(tensor_dy);
      const auto* data_ddx = tensor_ddx->data<T>();
      const DDim& dim = tensor_dy->dims();
      size_t N = static_cast<size_t>(product(dim));

      auto step = dim[dim.size() - 1];

      int s = -1;
      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_dy[i] = T(data_dout[s].real, -data_dout[s].imag) * data_ddx[i];
      }
    }

    if (tensor_ddout) {
      auto* data_ddout = ctx.template Alloc<T>(tensor_ddout);
      auto* data_x = tensor_x->data<T>();
      auto* data_y = tensor_y->data<T>();
      auto* data_ddx = tensor_ddx->data<T>();
      auto* data_ddy = tensor_ddy->data<T>();

      const DDim& dim = tensor_dy->dims();
      size_t N = static_cast<size_t>(product(dim));
      auto step = dim[dim.size() - 1];
      int s = -1;
      bool new_s = false;

      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) {
          ++s;
          new_s = true;
        }
        if (new_s) {
          data_ddout[s] = T(data_x[i].real, -data_x[i].imag) * data_ddy[i] +
                          T(data_y[i].real, -data_y[i].imag) * data_ddx[i];
        } else {
          data_ddout[s] += T(data_x[i].real, -data_x[i].imag) * data_ddy[i] +
                           T(data_y[i].real, -data_y[i].imag) * data_ddx[i];
        }
        new_s = false;
      }
    }
#endif
  }
};

template <typename DeviceContext, typename T>
struct DotDoubleGradFunction<DeviceContext, T, phi::funcs::DisableComplex<T>> {
  void operator()(const DeviceContext& ctx,
                  const DenseTensor* tensor_x,
                  const DenseTensor* tensor_y,
                  const DenseTensor* tensor_dout,
                  const DenseTensor* tensor_ddx,
                  const DenseTensor* tensor_ddy,
                  DenseTensor* tensor_dx,
                  DenseTensor* tensor_dy,
                  DenseTensor* tensor_ddout) {
#if defined(__NVCC__) || defined(__HIPCC__)
    if (1 == tensor_dout->dims().size()) {
      auto& dev = *ctx.eigen_device();
      auto dout = EigenVector<T>::Flatten(*tensor_dout);
      if (tensor_dx) {
        ctx.template Alloc<T>(tensor_dx);
        auto ddy = EigenVector<T>::Flatten(*tensor_ddy);
        Eigen::DSizes<int, 1> size(tensor_ddy->numel());
        auto dx = EigenVector<T>::Flatten(*tensor_dx);
        dx.device(dev) = ddy * dout.broadcast(size);
      }

      if (tensor_dy) {
        ctx.template Alloc<T>(tensor_dy);
        auto ddx = EigenVector<T>::Flatten(*tensor_ddx);
        Eigen::DSizes<int, 1> size(tensor_ddx->numel());

        auto dy = EigenVector<T>::Flatten(*tensor_dy);
        dy.device(dev) = ddx * dout.broadcast(size);
      }

      if (tensor_ddout) {
        ctx.template Alloc<T>(tensor_ddout);
        auto x = EigenVector<T>::Flatten(*tensor_x);
        auto y = EigenVector<T>::Flatten(*tensor_y);
        auto ddx = EigenVector<T>::Flatten(*tensor_ddx);
        auto ddy = EigenVector<T>::Flatten(*tensor_ddy);
        auto ddout = EigenVector<T>::Flatten(*tensor_ddout);
        ddout.device(dev) = (x * ddy + y * ddx).sum();
      }
    }
#else
    const auto* data_dout = tensor_dout->data<T>();

    if (tensor_dx) {
      auto* data_dx = ctx.template Alloc<T>(tensor_dx);
      const auto* data_ddy = tensor_ddy->data<T>();
      const DDim& dim = tensor_dx->dims();
      size_t N = static_cast<size_t>(product(dim));

      auto step = dim[dim.size() - 1];

      int s = -1;
      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_dx[i] = data_dout[s] * data_ddy[i];
      }
    }

    if (tensor_dy) {
      auto* data_dy = ctx.template Alloc<T>(tensor_dy);
      const auto* data_ddx = tensor_ddx->data<T>();
      const DDim& dim = tensor_dy->dims();
      size_t N = static_cast<size_t>(product(dim));

      auto step = dim[dim.size() - 1];

      int s = -1;
      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_dy[i] = data_dout[s] * data_ddx[i];
      }
    }

    if (tensor_ddout) {
      auto* data_ddout = ctx.template Alloc<T>(tensor_ddout);
      auto* data_x = tensor_x->data<T>();
      auto* data_y = tensor_y->data<T>();
      auto* data_ddx = tensor_ddx->data<T>();
      auto* data_ddy = tensor_ddy->data<T>();

      const DDim& dim = tensor_dy->dims();
      size_t N = static_cast<size_t>(product(dim));
      auto step = dim[dim.size() - 1];
      int s = -1;
      bool new_s = false;

      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) {
          ++s;
          new_s = true;
        }
        if (new_s) {
          data_ddout[s] = data_x[i] * data_ddy[i] + data_y[i] * data_ddx[i];
        } else {
          data_ddout[s] += data_x[i] * data_ddy[i] + data_y[i] * data_ddx[i];
        }
        new_s = false;
      }
    }
#endif
  }
};

template <typename DeviceContext, typename T, typename Enabel = void>
struct DotTripleGradFunction {
  void operator()(const DeviceContext& ctx,
                  const DenseTensor* in_tensor_x,
                  const DenseTensor* in_tensor_y,
                  const DenseTensor* in_tensor_ddx,
                  const DenseTensor* in_tensor_ddy,
                  const DenseTensor* in_tensor_d_dx,
                  const DenseTensor* in_tensor_d_dy,
                  const DenseTensor* in_tensor_dout,
                  const DenseTensor* in_tensor_d_ddout,
                  DenseTensor* out_tensor_d_x,
                  DenseTensor* out_tensor_d_y,
                  DenseTensor* out_tensor_d_dout,
                  DenseTensor* out_tensor_d_ddx,
                  DenseTensor* out_tensor_d_ddy);
};

// TODO(wuweilong): enable this function when the unittests framewark for multi
// grad is ok (dtype: complex64 or complex128).
template <typename DeviceContext, typename T>
struct DotTripleGradFunction<DeviceContext, T, phi::funcs::EnableComplex<T>> {
  void operator()(const DeviceContext& ctx,
                  const DenseTensor* in_tensor_x,
                  const DenseTensor* in_tensor_y,
                  const DenseTensor* in_tensor_ddx,
                  const DenseTensor* in_tensor_ddy,
                  const DenseTensor* in_tensor_d_dx,
                  const DenseTensor* in_tensor_d_dy,
                  const DenseTensor* in_tensor_dout,
                  const DenseTensor* in_tensor_d_ddout,
                  DenseTensor* out_tensor_d_x,
                  DenseTensor* out_tensor_d_y,
                  DenseTensor* out_tensor_d_dout,
                  DenseTensor* out_tensor_d_ddx,
                  DenseTensor* out_tensor_d_ddy) {
#if defined(__NVCC__) || defined(__HIPCC__)
    if (1 == in_tensor_d_ddout->dims().size()) {
      DenseTensor in_tensor_d_ddout_help;
      auto& dev = *ctx.eigen_device();
      if (out_tensor_d_x || out_tensor_d_y) {
        in_tensor_d_ddout_help =
            Conj<T, DeviceContext>(ctx, *in_tensor_d_ddout);
      }
      if (out_tensor_d_x) {
        auto ddy = EigenVector<T>::Flatten(*in_tensor_ddy);
        Eigen::DSizes<int, 1> size(in_tensor_ddy->numel());
        auto d_x = EigenVector<T>::Flatten(*out_tensor_d_x);
        auto d_ddout = EigenVector<T>::Flatten(in_tensor_d_ddout_help);
        d_x.device(dev) = ddy * d_ddout.broadcast(size);
      }

      if (out_tensor_d_y) {
        auto ddx = EigenVector<T>::Flatten(*in_tensor_ddx);
        Eigen::DSizes<int, 1> size(in_tensor_ddx->numel());
        auto d_y = EigenVector<T>::Flatten(*out_tensor_d_y);
        auto d_ddout = EigenVector<T>::Flatten(in_tensor_d_ddout_help);
        d_y.device(dev) = ddx * d_ddout.broadcast(size);
      }

      if (out_tensor_d_dout) {
        DenseTensor in_tensor_ddx_help =
            Conj<T, DeviceContext>(ctx, *in_tensor_ddx);
        DenseTensor in_tensor_ddy_help =
            Conj<T, DeviceContext>(ctx, *in_tensor_ddy);

        auto ddx = EigenVector<T>::Flatten(in_tensor_ddx_help);
        auto ddy = EigenVector<T>::Flatten(in_tensor_ddy_help);
        auto d_dx = EigenVector<T>::Flatten(*in_tensor_d_dx);
        auto d_dy = EigenVector<T>::Flatten(*in_tensor_d_dy);
        auto d_dout = EigenVector<T>::Flatten(*out_tensor_d_dout);
        d_dout.device(dev) = (ddx * d_dy + ddy * d_dx).sum();
      }

      if (out_tensor_d_ddx) {
        DenseTensor in_tensor_dout_help =
            Conj<T, DeviceContext>(ctx, *in_tensor_dout);
        DenseTensor in_tensor_y_help =
            Conj<T, DeviceContext>(ctx, *in_tensor_y);

        auto dout = EigenVector<T>::Flatten(in_tensor_dout_help);
        auto y = EigenVector<T>::Flatten(in_tensor_y_help);
        auto d_ddout = EigenVector<T>::Flatten(*in_tensor_d_ddout);
        auto d_dy = EigenVector<T>::Flatten(*in_tensor_d_dy);
        auto d_ddx = EigenVector<T>::Flatten(*out_tensor_d_ddx);
        Eigen::DSizes<int, 1> size(in_tensor_y->numel());
        d_ddx.device(dev) =
            (dout.broadcast(size) * d_dy + y * d_ddout.broadcast(size));
      }

      if (out_tensor_d_ddy) {
        DenseTensor in_tensor_dout_help =
            Conj<T, DeviceContext>(ctx, *in_tensor_dout);
        DenseTensor in_tensor_x_help =
            Conj<T, DeviceContext>(ctx, *in_tensor_x);

        auto dout = EigenVector<T>::Flatten(in_tensor_dout_help);
        auto x = EigenVector<T>::Flatten(in_tensor_x_help);
        auto d_ddout = EigenVector<T>::Flatten(*in_tensor_d_ddout);
        auto d_dx = EigenVector<T>::Flatten(*in_tensor_d_dx);
        auto d_ddy = EigenVector<T>::Flatten(*out_tensor_d_ddy);
        Eigen::DSizes<int, 1> size(in_tensor_x->numel());
        d_ddy.device(dev) =
            (dout.broadcast(size) * d_dx + x * d_ddout.broadcast(size));
      }
    }
#else
    const auto* data_d_ddout = in_tensor_d_ddout->data<T>();

    if (out_tensor_d_x) {
      auto* data_d_x = ctx.template Alloc<T>(out_tensor_d_x);
      const auto* data_ddy = in_tensor_ddy->data<T>();

      const DDim& dim = out_tensor_d_x->dims();
      size_t N = static_cast<size_t>(product(dim));
      auto step = dim[dim.size() - 1];
      int s = -1;

      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_d_x[i] = T(data_ddy[i].real, -data_ddy[i].imag) * data_d_ddout[s];
      }
    }

    if (out_tensor_d_y) {
      auto* data_d_y = ctx.template Alloc<T>(out_tensor_d_y);
      const auto* data_ddx = in_tensor_ddx->data<T>();

      const DDim& dim = out_tensor_d_y->dims();
      size_t N = static_cast<size_t>(product(dim));
      auto step = dim[dim.size() - 1];
      int s = -1;

      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_d_y[i] = T(data_ddx[i].real, -data_ddx[i].imag) * data_d_ddout[s];
      }
    }

    if (out_tensor_d_dout) {
      auto* data_d_dout = ctx.template Alloc<T>(out_tensor_d_dout);
      auto* data_ddx = in_tensor_ddx->data<T>();
      auto* data_ddy = in_tensor_ddy->data<T>();
      auto* data_d_dx = in_tensor_d_dx->data<T>();
      auto* data_d_dy = in_tensor_d_dy->data<T>();

      const DDim& dim = out_tensor_d_dout->dims();
      size_t N = static_cast<size_t>(product(dim));
      auto step = dim[dim.size() - 1];
      int s = -1;
      bool new_s = false;

      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) {
          ++s;
          new_s = true;
        }
        if (new_s) {
          data_d_dout[s] =
              T(data_ddy[i].real, -data_ddy[i].imag) * data_d_dx[i] +
              T(data_ddx[i].real, -data_ddx[i].imag) * data_d_dy[i];
        } else {
          data_d_dout[s] +=
              T(data_ddy[i].real, -data_ddy[i].imag) * data_d_dx[i] +
              T(data_ddx[i].real, -data_ddx[i].imag) * data_d_dy[i];
        }
        new_s = false;
      }
    }

    if (out_tensor_d_ddx) {
      auto* data_d_ddx = ctx.template Alloc<T>(out_tensor_d_ddx);
      auto* data_dout = in_tensor_dout->data<T>();
      auto* data_d_dy = in_tensor_d_dy->data<T>();
      auto* data_y = in_tensor_y->data<T>();
      auto* data_d_ddout = in_tensor_d_ddout->data<T>();

      const DDim& dim = out_tensor_d_ddx->dims();
      size_t N = static_cast<size_t>(product(dim));
      auto step = dim[dim.size() - 1];
      int s = -1;

      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_d_ddx[i] =
            T(data_dout[s].real, -data_dout[s].imag) * data_d_dy[i] +
            T(data_y[i].real, -data_y[i].imag) * data_d_ddout[s];
      }
    }

    if (out_tensor_d_ddy) {
      auto* data_d_ddy = ctx.template Alloc<T>(out_tensor_d_ddy);
      auto* data_dout = in_tensor_dout->data<T>();
      auto* data_d_dx = in_tensor_d_dx->data<T>();
      auto* data_x = in_tensor_x->data<T>();
      auto* data_d_ddout = in_tensor_d_ddout->data<T>();

      const DDim& dim = out_tensor_d_ddy->dims();
      size_t N = static_cast<size_t>(product(dim));
      auto step = dim[dim.size() - 1];
      int s = -1;

      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_d_ddy[i] =
            T(data_dout[s].real, -data_dout[s].imag) * data_d_dx[i] +
            T(data_x[i].real, -data_x[i].imag) * data_d_ddout[s];
      }
    }
#endif
  }
};

template <typename DeviceContext, typename T>
struct DotTripleGradFunction<DeviceContext, T, phi::funcs::DisableComplex<T>> {
  void operator()(const DeviceContext& ctx,
                  const DenseTensor* in_tensor_x,
                  const DenseTensor* in_tensor_y,
                  const DenseTensor* in_tensor_ddx,
                  const DenseTensor* in_tensor_ddy,
                  const DenseTensor* in_tensor_d_dx,
                  const DenseTensor* in_tensor_d_dy,
                  const DenseTensor* in_tensor_dout,
                  const DenseTensor* in_tensor_d_ddout,
                  DenseTensor* out_tensor_d_x,
                  DenseTensor* out_tensor_d_y,
                  DenseTensor* out_tensor_d_dout,
                  DenseTensor* out_tensor_d_ddx,
                  DenseTensor* out_tensor_d_ddy) {
#if defined(__NVCC__) || defined(__HIPCC__)
    if (1 == in_tensor_d_ddout->dims().size()) {
      auto& dev = *ctx.eigen_device();
      auto d_ddout = EigenVector<T>::Flatten(*in_tensor_d_ddout);
      if (out_tensor_d_x) {
        ctx.template Alloc<T>(out_tensor_d_x);
        auto ddy = EigenVector<T>::Flatten(*in_tensor_ddy);
        Eigen::DSizes<int, 1> size(in_tensor_ddy->numel());
        auto d_x = EigenVector<T>::Flatten(*out_tensor_d_x);
        d_x.device(dev) = ddy * d_ddout.broadcast(size);
      }

      if (out_tensor_d_y) {
        ctx.template Alloc<T>(out_tensor_d_y);
        auto ddx = EigenVector<T>::Flatten(*in_tensor_ddx);
        Eigen::DSizes<int, 1> size(in_tensor_ddx->numel());

        auto d_y = EigenVector<T>::Flatten(*out_tensor_d_y);
        d_y.device(dev) = ddx * d_ddout.broadcast(size);
      }

      if (out_tensor_d_dout) {
        ctx.template Alloc<T>(out_tensor_d_dout);
        auto ddx = EigenVector<T>::Flatten(*in_tensor_ddx);
        auto ddy = EigenVector<T>::Flatten(*in_tensor_ddy);
        auto d_dx = EigenVector<T>::Flatten(*in_tensor_d_dx);
        auto d_dy = EigenVector<T>::Flatten(*in_tensor_d_dy);
        auto d_dout = EigenVector<T>::Flatten(*out_tensor_d_dout);
        d_dout.device(dev) = (ddx * d_dy + ddy * d_dx).sum();
      }

      if (out_tensor_d_ddx) {
        ctx.template Alloc<T>(out_tensor_d_ddx);
        auto dout = EigenVector<T>::Flatten(*in_tensor_dout);
        auto y = EigenVector<T>::Flatten(*in_tensor_y);
        auto d_ddout = EigenVector<T>::Flatten(*in_tensor_d_ddout);
        auto d_dy = EigenVector<T>::Flatten(*in_tensor_d_dy);
        auto d_ddx = EigenVector<T>::Flatten(*out_tensor_d_ddx);
        Eigen::DSizes<int, 1> size(in_tensor_y->numel());
        d_ddx.device(dev) =
            (dout.broadcast(size) * d_dy + y * d_ddout.broadcast(size));
      }

      if (out_tensor_d_ddy) {
        ctx.template Alloc<T>(out_tensor_d_ddy);
        auto dout = EigenVector<T>::Flatten(*in_tensor_dout);
        auto x = EigenVector<T>::Flatten(*in_tensor_x);
        auto d_ddout = EigenVector<T>::Flatten(*in_tensor_d_ddout);
        auto d_dx = EigenVector<T>::Flatten(*in_tensor_d_dx);
        auto d_ddy = EigenVector<T>::Flatten(*out_tensor_d_ddy);
        Eigen::DSizes<int, 1> size(in_tensor_x->numel());
        d_ddy.device(dev) =
            (dout.broadcast(size) * d_dx + x * d_ddout.broadcast(size));
      }
    }
#else
    const auto* data_d_ddout = in_tensor_d_ddout->data<T>();

    if (out_tensor_d_x) {
      auto* data_d_x = ctx.template Alloc<T>(out_tensor_d_x);
      const auto* data_ddy = in_tensor_ddy->data<T>();

      const DDim& dim = out_tensor_d_x->dims();
      size_t N = static_cast<size_t>(product(dim));
      auto step = dim[dim.size() - 1];
      int s = -1;

      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_d_x[i] = data_ddy[i] * data_d_ddout[s];
      }
    }

    if (out_tensor_d_y) {
      auto* data_d_y = ctx.template Alloc<T>(out_tensor_d_y);
      const auto* data_ddx = in_tensor_ddx->data<T>();

      const DDim& dim = out_tensor_d_y->dims();
      size_t N = static_cast<size_t>(product(dim));
      auto step = dim[dim.size() - 1];
      int s = -1;

      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_d_y[i] = data_ddx[i] * data_d_ddout[s];
      }
    }

    if (out_tensor_d_dout) {
      auto* data_d_dout = ctx.template Alloc<T>(out_tensor_d_dout);
      auto* data_ddx = in_tensor_ddx->data<T>();
      auto* data_ddy = in_tensor_ddy->data<T>();
      auto* data_d_dx = in_tensor_d_dx->data<T>();
      auto* data_d_dy = in_tensor_d_dy->data<T>();

      const DDim& dim = in_tensor_ddx->dims();
      size_t N = static_cast<size_t>(product(dim));
      auto step = dim[dim.size() - 1];
      int s = -1;
      bool new_s = false;
      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) {
          ++s;
          new_s = true;
        }
        if (new_s) {
          data_d_dout[s] =
              data_ddy[i] * data_d_dx[i] + data_ddx[i] * data_d_dy[i];
        } else {
          data_d_dout[s] +=
              data_ddy[i] * data_d_dx[i] + data_ddx[i] * data_d_dy[i];
        }
        new_s = false;
      }
    }

    if (out_tensor_d_ddx) {
      auto* data_d_ddx = ctx.template Alloc<T>(out_tensor_d_ddx);
      auto* data_dout = in_tensor_dout->data<T>();
      auto* data_d_dy = in_tensor_d_dy->data<T>();
      auto* data_y = in_tensor_y->data<T>();
      auto* data_d_ddout = in_tensor_d_ddout->data<T>();

      const DDim& dim = out_tensor_d_ddx->dims();
      size_t N = static_cast<size_t>(product(dim));
      auto step = dim[dim.size() - 1];
      int s = -1;

      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_d_ddx[i] =
            data_dout[s] * data_d_dy[i] + data_y[i] * data_d_ddout[s];
      }
    }

    if (out_tensor_d_ddy) {
      auto* data_d_ddy = ctx.template Alloc<T>(out_tensor_d_ddy);
      auto* data_dout = in_tensor_dout->data<T>();
      auto* data_d_dx = in_tensor_d_dx->data<T>();
      auto* data_x = in_tensor_x->data<T>();
      auto* data_d_ddout = in_tensor_d_ddout->data<T>();

      const DDim& dim = out_tensor_d_ddy->dims();
      size_t N = static_cast<size_t>(product(dim));
      auto step = dim[dim.size() - 1];
      int s = -1;

      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_d_ddy[i] =
            data_dout[s] * data_d_dx[i] + data_x[i] * data_d_ddout[s];
      }
    }
#endif
  }
};

template <typename T, typename Context>
void DotGradKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   const DenseTensor& dout,
                   DenseTensor* dx,
                   DenseTensor* dy) {
  if (dx) {
    dev_ctx.template Alloc<T>(dx);
  }
  if (dy) {
    dev_ctx.template Alloc<T>(dy);
  }
  DotGradFunction<Context, T>()(dev_ctx, &x, &y, &dout, dx, dy);
}

template <typename T, typename Context>
void DotDoubleGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& y,
                         const DenseTensor& ddx,
                         const DenseTensor& ddy,
                         const DenseTensor& dout,
                         DenseTensor* dx,
                         DenseTensor* dy,
                         DenseTensor* ddout) {
  if (dx) {
    dev_ctx.template Alloc<T>(dx);
  }
  if (dy) {
    dev_ctx.template Alloc<T>(dy);
  }
  if (ddout) {
    dev_ctx.template Alloc<T>(ddout);
  }
  DotDoubleGradFunction<Context, T>()(
      dev_ctx, &x, &y, &dout, ddx, ddy, dx, dy, ddout);
}

template <typename T, typename Context>
void DotTripleGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& y,
                         const DenseTensor& ddx,
                         const DenseTensor& ddy,
                         const DenseTensor& d_dx,
                         const DenseTensor& d_dy,
                         const DenseTensor& dout,
                         const DenseTensor& d_ddout,
                         DenseTensor* d_x,
                         DenseTensor* d_y,
                         DenseTensor* d_ddx,
                         DenseTensor* d_ddy,
                         DenseTensor* d_dout) {
  if (d_x) {
    dev_ctx.template Alloc<T>(d_x);
  }
  if (d_y) {
    dev_ctx.template Alloc<T>(d_y);
  }
  if (d_ddx) {
    dev_ctx.template Alloc<T>(d_ddx);
  }
  if (d_ddy) {
    dev_ctx.template Alloc<T>(d_ddy);
  }
  if (d_dout) {
    dev_ctx.template Alloc<T>(d_dout);
  }

  DotTripleGradFunction<Context, T>()(dev_ctx,
                                      &x,
                                      &y,
                                      ddx,
                                      ddy,
                                      d_dx,
                                      d_dy,
                                      dout,
                                      d_ddout,
                                      d_x,
                                      d_y,
                                      d_dout,
                                      d_ddx,
                                      d_ddy);
}

}  // namespace phi
