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

#include "glog/logging.h"

#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
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
    VLOG(1) << "enable route";
#if defined(__NVCC__) || defined(__HIPCC__)
    if (1 >= tensor_dout->dims().size()) {
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
      size_t N = static_cast<size_t>(common::product(dim));

      auto _step = dim.size() > 0 ? dim[dim.size() - 1] : 1;
      auto step = _step != 0 ? _step : 1;

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
      size_t N = static_cast<size_t>(common::product(dim));

      auto _step = dim.size() > 0 ? dim[dim.size() - 1] : 1;
      auto step = _step != 0 ? _step : 1;

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
    if (1 >= tensor_dout->dims().size()) {
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
    auto const _B = d.size() == 0 ? 1 : d[d.size() - 1];
    auto const B = _B != 0 ? _B : 1;

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
                  const paddle::optional<DenseTensor>* tensor_ddx_opt,
                  const paddle::optional<DenseTensor>* tensor_ddy_opt,
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
                  const paddle::optional<DenseTensor>* tensor_ddx_opt,
                  const paddle::optional<DenseTensor>* tensor_ddy_opt,
                  DenseTensor* tensor_dx,
                  DenseTensor* tensor_dy,
                  DenseTensor* tensor_ddout) {
    const DenseTensor* tensor_ddx = tensor_ddx_opt->get_ptr();
    const DenseTensor* tensor_ddy = tensor_ddy_opt->get_ptr();
#if defined(__NVCC__) || defined(__HIPCC__)
    if (1 >= tensor_dout->dims().size()) {
      DenseTensor tensor_dout_help;
      auto& dev = *ctx.eigen_device();
      if (tensor_dx || tensor_dy) {
        tensor_dout_help = Conj<T, DeviceContext>(ctx, *tensor_dout);
      }
      if (tensor_dx && tensor_ddy) {
        ctx.template Alloc<T>(tensor_dx);
        auto ddy = EigenVector<T>::Flatten(*tensor_ddy);
        Eigen::DSizes<int, 1> size(tensor_ddy->numel());
        auto dx = EigenVector<T>::Flatten(*tensor_dx);
        auto dout = EigenVector<T>::Flatten(tensor_dout_help);
        dx.device(dev) = ddy * dout.broadcast(size);
      } else if (tensor_dx && !tensor_ddy) {
        FullLikeKernel<T, DeviceContext>(
            ctx, *tensor_x, Scalar(T(0.0, 0.0)), tensor_x->dtype(), tensor_dx);
      }

      if (tensor_dy && tensor_ddx) {
        ctx.template Alloc<T>(tensor_dy);
        auto ddx = EigenVector<T>::Flatten(*tensor_ddx);
        Eigen::DSizes<int, 1> size(tensor_ddx->numel());
        auto dy = EigenVector<T>::Flatten(*tensor_dy);
        auto dout = EigenVector<T>::Flatten(tensor_dout_help);
        dy.device(dev) = ddx * dout.broadcast(size);
      } else if (tensor_dy && !tensor_ddx) {
        FullLikeKernel<T, DeviceContext>(
            ctx, *tensor_y, Scalar(T(0.0, 0.0)), tensor_y->dtype(), tensor_dy);
      }

      if (tensor_ddout && tensor_ddx && tensor_ddy) {
        ctx.template Alloc<T>(tensor_ddout);
        DenseTensor tensor_x_help = Conj<T, DeviceContext>(ctx, *tensor_x);
        DenseTensor tensor_y_help = Conj<T, DeviceContext>(ctx, *tensor_y);

        auto x = EigenVector<T>::Flatten(tensor_x_help);
        auto y = EigenVector<T>::Flatten(tensor_y_help);
        auto ddx = EigenVector<T>::Flatten(*tensor_ddx);
        auto ddy = EigenVector<T>::Flatten(*tensor_ddy);
        auto ddout = EigenVector<T>::Flatten(*tensor_ddout);
        ddout.device(dev) = (x * ddy + y * ddx).sum();
      } else if (tensor_ddout && tensor_ddx && !tensor_ddy) {
        ctx.template Alloc<T>(tensor_ddout);
        DenseTensor tensor_y_help = Conj<T, DeviceContext>(ctx, *tensor_y);

        auto y = EigenVector<T>::Flatten(tensor_y_help);
        auto ddx = EigenVector<T>::Flatten(*tensor_ddx);
        auto ddout = EigenVector<T>::Flatten(*tensor_ddout);
        ddout.device(dev) = (y * ddx).sum();
      } else if (tensor_ddout && !tensor_ddx && tensor_ddy) {
        ctx.template Alloc<T>(tensor_ddout);
        DenseTensor tensor_x_help = Conj<T, DeviceContext>(ctx, *tensor_x);

        auto x = EigenVector<T>::Flatten(tensor_x_help);
        auto ddy = EigenVector<T>::Flatten(*tensor_ddy);
        auto ddout = EigenVector<T>::Flatten(*tensor_ddout);
        ddout.device(dev) = (x * ddy).sum();
      }
    }
#else
    const auto* data_dout = tensor_dout->data<T>();

    if (tensor_dx && tensor_ddy) {
      auto* data_dx = ctx.template Alloc<T>(tensor_dx);
      const auto* data_ddy = tensor_ddy->data<T>();
      const DDim& dim = tensor_dx->dims();
      size_t N = static_cast<size_t>(product(dim));

      auto _step = dim.size() > 0 ? dim[dim.size() - 1] : 1;
      auto step = _step != 0 ? _step : 1;

      int s = -1;
      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_dx[i] = T(data_dout[s].real, -data_dout[s].imag) * data_ddy[i];
      }
    } else if (tensor_dx && !tensor_ddy) {
      FullLikeKernel<T, DeviceContext>(
          ctx, *tensor_x, Scalar(T(0.0, 0.0)), tensor_x->dtype(), tensor_dx);
    }

    if (tensor_dy && tensor_ddx) {
      auto* data_dy = ctx.template Alloc<T>(tensor_dy);
      const auto* data_ddx = tensor_ddx->data<T>();
      const DDim& dim = tensor_dy->dims();
      size_t N = static_cast<size_t>(product(dim));

      auto _step = dim.size() > 0 ? dim[dim.size() - 1] : 1;
      auto step = _step != 0 ? _step : 1;

      int s = -1;
      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_dy[i] = T(data_dout[s].real, -data_dout[s].imag) * data_ddx[i];
      }
    } else if (tensor_dy && !tensor_ddx) {
      FullLikeKernel<T, DeviceContext>(
          ctx, *tensor_y, Scalar(T(0.0, 0.0)), tensor_y->dtype(), tensor_dy);
    }

    if (tensor_ddout && tensor_ddx && tensor_ddy) {
      auto* data_ddout = ctx.template Alloc<T>(tensor_ddout);
      auto* data_x = tensor_x->data<T>();
      auto* data_y = tensor_y->data<T>();
      auto* data_ddx = tensor_ddx->data<T>();
      auto* data_ddy = tensor_ddy->data<T>();

      const DDim& dim = tensor_dy->dims();
      size_t N = static_cast<size_t>(product(dim));
      auto _step = dim.size() > 0 ? dim[dim.size() - 1] : 1;
      auto step = _step != 0 ? _step : 1;
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
    } else if (tensor_ddout && tensor_ddx && !tensor_ddy) {
      auto* data_ddout = ctx.template Alloc<T>(tensor_ddout);
      auto* data_y = tensor_y->data<T>();
      auto* data_ddx = tensor_ddx->data<T>();

      const DDim& dim = tensor_dy->dims();
      size_t N = static_cast<size_t>(product(dim));
      auto _step = dim.size() > 0 ? dim[dim.size() - 1] : 1;
      auto step = _step != 0 ? _step : 1;
      int s = -1;
      bool new_s = false;

      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) {
          ++s;
          new_s = true;
        }
        if (new_s) {
          data_ddout[s] = T(data_y[i].real, -data_y[i].imag) * data_ddx[i];
        } else {
          data_ddout[s] += T(data_y[i].real, -data_y[i].imag) * data_ddx[i];
        }
        new_s = false;
      }
    } else if (tensor_ddout && !tensor_ddx && tensor_ddy) {
      auto* data_ddout = ctx.template Alloc<T>(tensor_ddout);
      auto* data_x = tensor_x->data<T>();
      auto* data_ddy = tensor_ddy->data<T>();

      const DDim& dim = tensor_dx->dims();
      size_t N = static_cast<size_t>(product(dim));
      auto _step = dim.size() > 0 ? dim[dim.size() - 1] : 1;
      auto step = _step != 0 ? _step : 1;
      int s = -1;
      bool new_s = false;

      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) {
          ++s;
          new_s = true;
        }
        if (new_s) {
          data_ddout[s] = T(data_x[i].real, -data_x[i].imag) * data_ddy[i];
        } else {
          data_ddout[s] += T(data_x[i].real, -data_x[i].imag) * data_ddy[i];
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
                  const paddle::optional<DenseTensor>* tensor_ddx_opt,
                  const paddle::optional<DenseTensor>* tensor_ddy_opt,
                  DenseTensor* tensor_dx,
                  DenseTensor* tensor_dy,
                  DenseTensor* tensor_ddout) {
    const DenseTensor* tensor_ddx = tensor_ddx_opt->get_ptr();
    const DenseTensor* tensor_ddy = tensor_ddy_opt->get_ptr();
#if defined(__NVCC__) || defined(__HIPCC__)
    if (1 >= tensor_dout->dims().size()) {
      auto& dev = *ctx.eigen_device();
      auto x = EigenVector<T>::Flatten(*tensor_x);
      auto y = EigenVector<T>::Flatten(*tensor_y);
      auto dout = EigenVector<T>::Flatten(*tensor_dout);
      if (tensor_dx && tensor_ddy) {
        ctx.template Alloc<T>(tensor_dx);
        auto ddy = EigenVector<T>::Flatten(*tensor_ddy);
        Eigen::DSizes<int, 1> size(tensor_ddy->numel());
        auto dx = EigenVector<T>::Flatten(*tensor_dx);
        dx.device(dev) = ddy * dout.broadcast(size);
      } else if (tensor_dx && !tensor_ddy) {
        FullLikeKernel<T, DeviceContext>(
            ctx, *tensor_x, Scalar(0.0), tensor_x->dtype(), tensor_dx);
      }

      if (tensor_dy && tensor_ddx) {
        ctx.template Alloc<T>(tensor_dy);
        auto ddx = EigenVector<T>::Flatten(*tensor_ddx);
        Eigen::DSizes<int, 1> size(tensor_ddx->numel());
        auto dy = EigenVector<T>::Flatten(*tensor_dy);
        dy.device(dev) = ddx * dout.broadcast(size);
      } else if (tensor_dy && !tensor_ddx) {
        FullLikeKernel<T, DeviceContext>(
            ctx, *tensor_y, Scalar(0.0), tensor_y->dtype(), tensor_dy);
      }

      if (tensor_ddout && tensor_ddx && tensor_ddy) {
        ctx.template Alloc<T>(tensor_ddout);
        auto ddx = EigenVector<T>::Flatten(*tensor_ddx);
        auto ddy = EigenVector<T>::Flatten(*tensor_ddy);
        auto ddout = EigenVector<T>::Flatten(*tensor_ddout);
        ddout.device(dev) = (x * ddy + y * ddx).sum();
      } else if (tensor_ddout && tensor_ddx && !tensor_ddy) {
        ctx.template Alloc<T>(tensor_ddout);
        auto ddx = EigenVector<T>::Flatten(*tensor_ddx);
        auto ddout = EigenVector<T>::Flatten(*tensor_ddout);
        ddout.device(dev) = (y * ddx).sum();
      } else if (tensor_ddout && !tensor_ddx && tensor_ddy) {
        ctx.template Alloc<T>(tensor_ddout);
        auto ddy = EigenVector<T>::Flatten(*tensor_ddy);
        auto ddout = EigenVector<T>::Flatten(*tensor_ddout);
        ddout.device(dev) = (x * ddy).sum();
      }
    }
#else
    const T* data_x = tensor_x->data<T>();
    const T* data_y = tensor_y->data<T>();
    const T* data_dout = tensor_dout->data<T>();
    const T* data_ddx = tensor_ddx ? tensor_ddx->data<T>() : nullptr;
    const T* data_ddy = tensor_ddy ? tensor_ddy->data<T>() : nullptr;
    if (tensor_dx && tensor_ddy) {
      auto* data_dx = ctx.template Alloc<T>(tensor_dx);
      const DDim& dim = tensor_dx->dims();
      size_t N = static_cast<size_t>(product(dim));
      auto _step = dim.size() > 0 ? dim[dim.size() - 1] : 1;
      auto step = _step != 0 ? _step : 1;
      int s = -1;
      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_dx[i] = data_dout[s] * data_ddy[i];
      }
    } else if (tensor_dx && !tensor_ddy) {
      FullLikeKernel<T, DeviceContext>(
          ctx, *tensor_x, Scalar(0.0), tensor_x->dtype(), tensor_dx);
    }

    if (tensor_dy && tensor_ddx) {
      auto* data_dy = ctx.template Alloc<T>(tensor_dy);
      const DDim& dim = tensor_dy->dims();
      size_t N = static_cast<size_t>(product(dim));
      auto _step = dim.size() > 0 ? dim[dim.size() - 1] : 1;
      auto step = _step != 0 ? _step : 1;
      int s = -1;
      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_dy[i] = data_dout[s] * data_ddx[i];
      }
    } else if (tensor_dy) {
      FullLikeKernel<T, DeviceContext>(
          ctx, *tensor_y, Scalar(0.0), tensor_y->dtype(), tensor_dy);
    }

    if (tensor_ddout && tensor_ddx && tensor_ddy) {
      auto* data_ddout = ctx.template Alloc<T>(tensor_ddout);
      const DDim& dim = tensor_dy->dims();
      size_t N = static_cast<size_t>(product(dim));
      auto _step = dim.size() > 0 ? dim[dim.size() - 1] : 1;
      auto step = _step != 0 ? _step : 1;
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
    } else if (tensor_ddout && tensor_ddx && !tensor_ddy) {
      auto* data_ddout = ctx.template Alloc<T>(tensor_ddout);
      const DDim& dim = tensor_dy->dims();
      size_t N = static_cast<size_t>(product(dim));
      auto _step = dim.size() > 0 ? dim[dim.size() - 1] : 1;
      auto step = _step != 0 ? _step : 1;
      int s = -1;
      bool new_s = false;
      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) {
          ++s;
          new_s = true;
        }
        if (new_s) {
          data_ddout[s] = data_y[i] * data_ddx[i];
        } else {
          data_ddout[s] += data_y[i] * data_ddx[i];
        }
        new_s = false;
      }
    } else if (tensor_ddout && !tensor_ddx && tensor_ddy) {
      auto* data_ddout = ctx.template Alloc<T>(tensor_ddout);
      const DDim& dim = tensor_dx->dims();
      size_t N = static_cast<size_t>(product(dim));
      auto _step = dim.size() > 0 ? dim[dim.size() - 1] : 1;
      auto step = _step != 0 ? _step : 1;
      int s = -1;
      bool new_s = false;
      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) {
          ++s;
          new_s = true;
        }
        if (new_s) {
          data_ddout[s] = data_x[i] * data_ddy[i];
        } else {
          data_ddout[s] += data_x[i] * data_ddy[i];
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
                  const DenseTensor* in_tensor_dout,
                  const paddle::optional<DenseTensor>* in_tensor_ddx_opt,
                  const paddle::optional<DenseTensor>* in_tensor_ddy_opt,
                  const paddle::optional<DenseTensor>* in_tensor_d_dx_opt,
                  const paddle::optional<DenseTensor>* in_tensor_d_dy_opt,
                  const paddle::optional<DenseTensor>* in_tensor_d_ddout_opt,
                  DenseTensor* out_tensor_d_x,
                  DenseTensor* out_tensor_d_y,
                  DenseTensor* out_tensor_d_dout,
                  DenseTensor* out_tensor_d_ddx,
                  DenseTensor* out_tensor_d_ddy);
};

// TODO(wuweilong): enable this function when the unittest framework for multi
// grad is ok (dtype: complex64 or complex128).
template <typename DeviceContext, typename T>
struct DotTripleGradFunction<DeviceContext, T, phi::funcs::EnableComplex<T>> {
  void operator()(const DeviceContext& ctx,
                  const DenseTensor* in_tensor_x,
                  const DenseTensor* in_tensor_y,
                  const DenseTensor* in_tensor_dout,
                  const paddle::optional<DenseTensor>* in_tensor_ddx_opt,
                  const paddle::optional<DenseTensor>* in_tensor_ddy_opt,
                  const paddle::optional<DenseTensor>* in_tensor_d_dx_opt,
                  const paddle::optional<DenseTensor>* in_tensor_d_dy_opt,
                  const paddle::optional<DenseTensor>* in_tensor_d_ddout_opt,
                  DenseTensor* out_tensor_d_x,
                  DenseTensor* out_tensor_d_y,
                  DenseTensor* out_tensor_d_dout,
                  DenseTensor* out_tensor_d_ddx,
                  DenseTensor* out_tensor_d_ddy) {
    const DenseTensor* in_tensor_ddx = in_tensor_ddx_opt->get_ptr();
    const DenseTensor* in_tensor_ddy = in_tensor_ddy_opt->get_ptr();
    const DenseTensor* in_tensor_d_dx = in_tensor_d_dx_opt->get_ptr();
    const DenseTensor* in_tensor_d_dy = in_tensor_d_dy_opt->get_ptr();
    const DenseTensor* in_tensor_d_ddout = in_tensor_d_ddout_opt->get_ptr();
#if defined(__NVCC__) || defined(__HIPCC__)
    if (1 >= in_tensor_dout->dims().size()) {
      auto& dev = *ctx.eigen_device();
      DenseTensor in_tensor_x_help = Conj<T, DeviceContext>(ctx, *in_tensor_x);
      DenseTensor in_tensor_y_help = Conj<T, DeviceContext>(ctx, *in_tensor_y);
      DenseTensor in_tensor_dout_help =
          Conj<T, DeviceContext>(ctx, *in_tensor_dout);
      DenseTensor in_tensor_ddx_help;
      DenseTensor in_tensor_ddy_help;
      if (in_tensor_ddx) {
        in_tensor_ddx_help = Conj<T, DeviceContext>(ctx, *in_tensor_ddx);
      }
      if (in_tensor_ddy) {
        in_tensor_ddy_help = Conj<T, DeviceContext>(ctx, *in_tensor_ddy);
      }

      bool d_dout_flag = false;
      bool d_ddx_flag = false;
      bool d_ddy_flag = false;

      if (in_tensor_ddx) {
        if (out_tensor_d_y && in_tensor_d_ddout) {
          ctx.template Alloc<T>(out_tensor_d_y);
          auto ddx = EigenVector<T>::Flatten(in_tensor_ddx_help);
          Eigen::DSizes<int, 1> size(in_tensor_ddx->numel());
          auto d_y = EigenVector<T>::Flatten(*out_tensor_d_y);
          auto d_ddout = EigenVector<T>::Flatten(*in_tensor_d_ddout);
          d_y.device(dev) = ddx * d_ddout.broadcast(size);
        }
        if (out_tensor_d_dout && in_tensor_d_dy) {
          ctx.template Alloc<T>(out_tensor_d_dout);
          auto ddx = EigenVector<T>::Flatten(in_tensor_ddx_help);
          auto d_dy = EigenVector<T>::Flatten(*in_tensor_d_dy);
          auto d_dout = EigenVector<T>::Flatten(*out_tensor_d_dout);
          d_dout.device(dev) = (ddx * d_dy).sum();
          d_dout_flag = true;
        }
      }

      if (in_tensor_ddy) {
        if (out_tensor_d_x && in_tensor_d_ddout) {
          ctx.template Alloc<T>(out_tensor_d_x);
          auto ddy = EigenVector<T>::Flatten(in_tensor_ddy_help);
          Eigen::DSizes<int, 1> size(in_tensor_ddy->numel());
          auto d_x = EigenVector<T>::Flatten(*out_tensor_d_x);
          auto d_ddout = EigenVector<T>::Flatten(*in_tensor_d_ddout);
          d_x.device(dev) = ddy * d_ddout.broadcast(size);
        }
        if (out_tensor_d_dout && in_tensor_d_dx) {
          ctx.template Alloc<T>(out_tensor_d_dout);
          auto ddy = EigenVector<T>::Flatten(in_tensor_ddy_help);
          auto d_dx = EigenVector<T>::Flatten(*in_tensor_d_dx);
          auto d_dout = EigenVector<T>::Flatten(*out_tensor_d_dout);
          if (d_dout_flag) {
            d_dout.device(dev) += (ddy * d_dx).sum();
          } else {
            d_dout.device(dev) = (ddy * d_dx).sum();
          }
        }
      }

      if (in_tensor_d_dx) {
        if (out_tensor_d_ddy) {
          ctx.template Alloc<T>(out_tensor_d_ddy);
          auto dout = EigenVector<T>::Flatten(in_tensor_dout_help);
          auto d_dx = EigenVector<T>::Flatten(*in_tensor_d_dx);
          auto d_ddy = EigenVector<T>::Flatten(*out_tensor_d_ddy);
          Eigen::DSizes<int, 1> size(in_tensor_x->numel());
          d_ddy.device(dev) = (dout.broadcast(size) * d_dx);
          d_ddy_flag = true;
        }
      }

      if (in_tensor_d_dy) {
        if (out_tensor_d_ddx) {
          ctx.template Alloc<T>(out_tensor_d_ddx);
          auto dout = EigenVector<T>::Flatten(in_tensor_dout_help);
          auto d_dy = EigenVector<T>::Flatten(*in_tensor_d_dy);
          auto d_ddx = EigenVector<T>::Flatten(*out_tensor_d_ddx);
          Eigen::DSizes<int, 1> size(in_tensor_y->numel());
          d_ddx.device(dev) = (dout.broadcast(size) * d_dy);
          d_ddx_flag = true;
        }
      }

      if (in_tensor_d_ddout) {
        if (out_tensor_d_ddx) {
          ctx.template Alloc<T>(out_tensor_d_ddx);
          auto y = EigenVector<T>::Flatten(in_tensor_y_help);
          auto d_ddout = EigenVector<T>::Flatten(*in_tensor_d_ddout);
          Eigen::DSizes<int, 1> size(in_tensor_y->numel());
          auto d_ddx = EigenVector<T>::Flatten(*out_tensor_d_ddx);
          if (d_ddx_flag) {
            d_ddx.device(dev) += (y * d_ddout.broadcast(size));
          } else {
            d_ddx.device(dev) = (y * d_ddout.broadcast(size));
          }
        }
        if (out_tensor_d_ddy) {
          ctx.template Alloc<T>(out_tensor_d_ddy);
          auto x = EigenVector<T>::Flatten(in_tensor_x_help);
          auto d_ddout = EigenVector<T>::Flatten(*in_tensor_d_ddout);
          Eigen::DSizes<int, 1> size(in_tensor_x->numel());
          auto d_ddy = EigenVector<T>::Flatten(*out_tensor_d_ddy);
          if (d_ddy_flag) {
            d_ddy.device(dev) += (x * d_ddout.broadcast(size));
          } else {
            d_ddy.device(dev) = (x * d_ddout.broadcast(size));
          }
        }
      }
      if (out_tensor_d_x && !out_tensor_d_x->IsInitialized()) {
        FullLikeKernel<T, DeviceContext>(ctx,
                                         *in_tensor_x,
                                         Scalar(T(0.0, 0.0)),
                                         in_tensor_x->dtype(),
                                         out_tensor_d_x);
      }
      if (out_tensor_d_y && !out_tensor_d_y->IsInitialized()) {
        FullLikeKernel<T, DeviceContext>(ctx,
                                         *in_tensor_y,
                                         Scalar(T(0.0, 0.0)),
                                         in_tensor_y->dtype(),
                                         out_tensor_d_y);
      }
      if (out_tensor_d_dout && !out_tensor_d_dout->IsInitialized()) {
        FullLikeKernel<T, DeviceContext>(ctx,
                                         *in_tensor_dout,
                                         Scalar(T(0.0, 0.0)),
                                         in_tensor_dout->dtype(),
                                         out_tensor_d_dout);
      }
      if (out_tensor_d_ddx && !out_tensor_d_ddx->IsInitialized()) {
        FullLikeKernel<T, DeviceContext>(ctx,
                                         *in_tensor_x,
                                         Scalar(T(0.0, 0.0)),
                                         in_tensor_x->dtype(),
                                         out_tensor_d_ddx);
      }
      if (out_tensor_d_ddy && !out_tensor_d_ddy->IsInitialized()) {
        FullLikeKernel<T, DeviceContext>(ctx,
                                         *in_tensor_y,
                                         Scalar(T(0.0, 0.0)),
                                         in_tensor_y->dtype(),
                                         out_tensor_d_ddy);
      }
    }
#else
    const T* data_x = in_tensor_x->data<T>();
    const T* data_y = in_tensor_y->data<T>();
    const T* data_dout = in_tensor_dout->data<T>();
    const T* data_ddx = in_tensor_ddx ? in_tensor_ddx->data<T>() : nullptr;
    const T* data_ddy = in_tensor_ddy ? in_tensor_ddy->data<T>() : nullptr;
    const T* data_d_dx = in_tensor_d_dx ? in_tensor_d_dx->data<T>() : nullptr;
    const T* data_d_dy = in_tensor_d_dy ? in_tensor_d_dy->data<T>() : nullptr;
    const T* data_d_ddout =
        in_tensor_d_ddout ? in_tensor_d_ddout->data<T>() : nullptr;

    bool d_dout_flag = false;
    bool d_ddx_flag = false;
    bool d_ddy_flag = false;

    if (data_ddx) {
      if (out_tensor_d_y && data_d_ddout) {
        auto* data_d_y = ctx.template Alloc<T>(out_tensor_d_y);
        const DDim& dim = out_tensor_d_y->dims();
        size_t N = static_cast<size_t>(product(dim));
        auto _step = dim.size() > 0 ? dim[dim.size() - 1] : 1;
        auto step = _step != 0 ? _step : 1;
        int s = -1;

        for (size_t i = 0; i < N; ++i) {
          if (0 == i % step) ++s;
          data_d_y[i] =
              T(data_ddx[i].real, -data_ddx[i].imag) * data_d_ddout[s];
        }
      }

      if (out_tensor_d_dout && data_d_dy) {
        auto* data_d_dout = ctx.template Alloc<T>(out_tensor_d_dout);
        const DDim& dim = in_tensor_x->dims();
        size_t N = static_cast<size_t>(product(dim));
        auto _step = dim.size() > 0 ? dim[dim.size() - 1] : 1;
        auto step = _step != 0 ? _step : 1;
        int s = -1;
        bool new_s = false;
        for (size_t i = 0; i < N; ++i) {
          if (0 == i % step) {
            ++s;
            new_s = true;
          }
          if (new_s) {
            data_d_dout[s] =
                T(data_ddx[i].real, -data_ddx[i].imag) * data_d_dy[i];
          } else {
            data_d_dout[s] +=
                T(data_ddx[i].real, -data_ddx[i].imag) * data_d_dy[i];
          }
          new_s = false;
        }
        d_dout_flag = true;
      }
    }

    if (data_ddy) {
      if (out_tensor_d_x && data_d_ddout) {
        auto* data_d_x = ctx.template Alloc<T>(out_tensor_d_x);
        const DDim& dim = out_tensor_d_x->dims();
        size_t N = static_cast<size_t>(product(dim));
        auto _step = dim.size() > 0 ? dim[dim.size() - 1] : 1;
        auto step = _step != 0 ? _step : 1;
        int s = -1;

        for (size_t i = 0; i < N; ++i) {
          if (0 == i % step) ++s;
          data_d_x[i] =
              T(data_ddy[i].real, -data_ddy[i].imag) * data_d_ddout[s];
        }
      }
      if (out_tensor_d_dout && data_d_dx) {
        auto* data_d_dout = ctx.template Alloc<T>(out_tensor_d_dout);
        const DDim& dim = in_tensor_x->dims();
        size_t N = static_cast<size_t>(product(dim));
        auto _step = dim.size() > 0 ? dim[dim.size() - 1] : 1;
        auto step = _step != 0 ? _step : 1;
        int s = -1;
        bool new_s = false;
        if (d_dout_flag) {
          for (size_t i = 0; i < N; ++i) {
            if (0 == i % step) {
              ++s;
            }
            data_d_dout[s] +=
                T(data_ddy[i].real, -data_ddy[i].imag) * data_d_dx[i];
          }
        } else {
          for (size_t i = 0; i < N; ++i) {
            if (0 == i % step) {
              ++s;
              new_s = true;
            }
            if (new_s) {
              data_d_dout[s] =
                  T(data_ddy[i].real, -data_ddy[i].imag) * data_d_dx[i];
            } else {
              data_d_dout[s] +=
                  T(data_ddy[i].real, -data_ddy[i].imag) * data_d_dx[i];
            }
            new_s = false;
          }
        }
      }
    }

    if (data_d_dx) {
      if (out_tensor_d_ddy) {
        auto* data_d_ddy = ctx.template Alloc<T>(out_tensor_d_ddy);
        const DDim& dim = out_tensor_d_ddy->dims();
        size_t N = static_cast<size_t>(product(dim));
        auto _step = dim.size() > 0 ? dim[dim.size() - 1] : 1;
        auto step = _step != 0 ? _step : 1;
        int s = -1;
        for (size_t i = 0; i < N; ++i) {
          if (0 == i % step) ++s;
          data_d_ddy[i] =
              T(data_dout[s].real, -data_dout[s].imag) * data_d_dx[i];
        }
        d_ddy_flag = true;
      }
    }

    if (data_d_dy) {
      if (out_tensor_d_ddx) {
        auto* data_d_ddx = ctx.template Alloc<T>(out_tensor_d_ddx);
        const DDim& dim = out_tensor_d_ddx->dims();
        size_t N = static_cast<size_t>(product(dim));
        auto _step = dim.size() > 0 ? dim[dim.size() - 1] : 1;
        auto step = _step != 0 ? _step : 1;
        int s = -1;
        for (size_t i = 0; i < N; ++i) {
          if (0 == i % step) ++s;
          data_d_ddx[i] =
              T(data_dout[s].real, -data_dout[s].imag) * data_d_dy[i];
        }
      }
      d_ddx_flag = true;
    }

    if (data_d_ddout) {
      if (out_tensor_d_ddx) {
        auto* data_d_ddx = ctx.template Alloc<T>(out_tensor_d_ddx);
        const DDim& dim = out_tensor_d_ddx->dims();
        size_t N = static_cast<size_t>(product(dim));
        auto _step = dim.size() > 0 ? dim[dim.size() - 1] : 1;
        auto step = _step != 0 ? _step : 1;
        int s = -1;
        if (d_ddx_flag) {
          for (size_t i = 0; i < N; ++i) {
            if (0 == i % step) ++s;
            data_d_ddx[i] +=
                T(data_y[i].real, -data_y[i].imag) * data_d_ddout[s];
          }
        } else {
          for (size_t i = 0; i < N; ++i) {
            if (0 == i % step) ++s;
            data_d_ddx[i] =
                T(data_y[i].real, -data_y[i].imag) * data_d_ddout[s];
          }
        }
      }
      if (out_tensor_d_ddy) {
        auto* data_d_ddy = ctx.template Alloc<T>(out_tensor_d_ddy);
        const DDim& dim = out_tensor_d_ddy->dims();
        size_t N = static_cast<size_t>(product(dim));
        auto _step = dim.size() > 0 ? dim[dim.size() - 1] : 1;
        auto step = _step != 0 ? _step : 1;
        int s = -1;
        if (d_ddy_flag) {
          for (size_t i = 0; i < N; ++i) {
            if (0 == i % step) ++s;
            data_d_ddy[i] +=
                T(data_x[i].real, -data_x[i].imag) * data_d_ddout[s];
          }
        } else {
          for (size_t i = 0; i < N; ++i) {
            if (0 == i % step) ++s;
            data_d_ddy[i] =
                T(data_x[i].real, -data_x[i].imag) * data_d_ddout[s];
          }
        }
      }
    }

    if (out_tensor_d_x && !out_tensor_d_x->IsInitialized()) {
      FullLikeKernel<T, DeviceContext>(ctx,
                                       *in_tensor_x,
                                       Scalar(T(0.0, 0.0)),
                                       in_tensor_x->dtype(),
                                       out_tensor_d_x);
    }
    if (out_tensor_d_y && !out_tensor_d_y->IsInitialized()) {
      FullLikeKernel<T, DeviceContext>(ctx,
                                       *in_tensor_y,
                                       Scalar(T(0.0, 0.0)),
                                       in_tensor_y->dtype(),
                                       out_tensor_d_y);
    }
    if (out_tensor_d_dout && !out_tensor_d_dout->IsInitialized()) {
      FullLikeKernel<T, DeviceContext>(ctx,
                                       *in_tensor_dout,
                                       Scalar(T(0.0, 0.0)),
                                       in_tensor_dout->dtype(),
                                       out_tensor_d_dout);
    }
    if (out_tensor_d_ddx && !out_tensor_d_ddx->IsInitialized()) {
      FullLikeKernel<T, DeviceContext>(ctx,
                                       *in_tensor_x,
                                       Scalar(T(0.0, 0.0)),
                                       in_tensor_x->dtype(),
                                       out_tensor_d_ddx);
    }
    if (out_tensor_d_ddy && !out_tensor_d_ddy->IsInitialized()) {
      FullLikeKernel<T, DeviceContext>(ctx,
                                       *in_tensor_y,
                                       Scalar(T(0.0, 0.0)),
                                       in_tensor_y->dtype(),
                                       out_tensor_d_ddy);
    }

#endif
  }
};

template <typename DeviceContext, typename T>
struct DotTripleGradFunction<DeviceContext, T, phi::funcs::DisableComplex<T>> {
  void operator()(const DeviceContext& ctx,
                  const DenseTensor* in_tensor_x,
                  const DenseTensor* in_tensor_y,
                  const DenseTensor* in_tensor_dout,
                  const paddle::optional<DenseTensor>* in_tensor_ddx_opt,
                  const paddle::optional<DenseTensor>* in_tensor_ddy_opt,
                  const paddle::optional<DenseTensor>* in_tensor_d_dx_opt,
                  const paddle::optional<DenseTensor>* in_tensor_d_dy_opt,
                  const paddle::optional<DenseTensor>* in_tensor_d_ddout_opt,
                  DenseTensor* out_tensor_d_x,
                  DenseTensor* out_tensor_d_y,
                  DenseTensor* out_tensor_d_dout,
                  DenseTensor* out_tensor_d_ddx,
                  DenseTensor* out_tensor_d_ddy) {
    const DenseTensor* in_tensor_ddx = in_tensor_ddx_opt->get_ptr();
    const DenseTensor* in_tensor_ddy = in_tensor_ddy_opt->get_ptr();
    const DenseTensor* in_tensor_d_dx = in_tensor_d_dx_opt->get_ptr();
    const DenseTensor* in_tensor_d_dy = in_tensor_d_dy_opt->get_ptr();
    const DenseTensor* in_tensor_d_ddout = in_tensor_d_ddout_opt->get_ptr();
#if defined(__NVCC__) || defined(__HIPCC__)
    if (1 >= in_tensor_dout->dims().size()) {
      auto& dev = *ctx.eigen_device();
      bool d_dout_flag = false;
      bool d_ddx_flag = false;
      bool d_ddy_flag = false;

      if (in_tensor_ddx) {
        if (out_tensor_d_y && in_tensor_d_ddout) {
          ctx.template Alloc<T>(out_tensor_d_y);
          auto ddx = EigenVector<T>::Flatten(*in_tensor_ddx);
          Eigen::DSizes<int, 1> size(in_tensor_ddx->numel());
          auto d_y = EigenVector<T>::Flatten(*out_tensor_d_y);
          auto d_ddout = EigenVector<T>::Flatten(*in_tensor_d_ddout);
          d_y.device(dev) = ddx * d_ddout.broadcast(size);
        }
        if (out_tensor_d_dout && in_tensor_d_dy) {
          ctx.template Alloc<T>(out_tensor_d_dout);
          auto ddx = EigenVector<T>::Flatten(*in_tensor_ddx);
          auto d_dy = EigenVector<T>::Flatten(*in_tensor_d_dy);
          auto d_dout = EigenVector<T>::Flatten(*out_tensor_d_dout);
          d_dout.device(dev) = (ddx * d_dy).sum();
          d_dout_flag = true;
        }
      }

      if (in_tensor_ddy) {
        if (out_tensor_d_x && in_tensor_d_ddout) {
          ctx.template Alloc<T>(out_tensor_d_x);
          auto ddy = EigenVector<T>::Flatten(*in_tensor_ddy);
          Eigen::DSizes<int, 1> size(in_tensor_ddy->numel());
          auto d_x = EigenVector<T>::Flatten(*out_tensor_d_x);
          auto d_ddout = EigenVector<T>::Flatten(*in_tensor_d_ddout);
          d_x.device(dev) = ddy * d_ddout.broadcast(size);
        }
        if (out_tensor_d_dout && in_tensor_d_dx) {
          ctx.template Alloc<T>(out_tensor_d_dout);
          auto ddy = EigenVector<T>::Flatten(*in_tensor_ddy);
          auto d_dx = EigenVector<T>::Flatten(*in_tensor_d_dx);
          auto d_dout = EigenVector<T>::Flatten(*out_tensor_d_dout);
          if (d_dout_flag) {
            d_dout.device(dev) += (ddy * d_dx).sum();
          } else {
            d_dout.device(dev) = (ddy * d_dx).sum();
          }
        }
      }

      if (in_tensor_d_dx) {
        if (out_tensor_d_ddy) {
          ctx.template Alloc<T>(out_tensor_d_ddy);
          auto dout = EigenVector<T>::Flatten(*in_tensor_dout);
          auto d_dx = EigenVector<T>::Flatten(*in_tensor_d_dx);
          auto d_ddy = EigenVector<T>::Flatten(*out_tensor_d_ddy);
          Eigen::DSizes<int, 1> size(in_tensor_x->numel());
          d_ddy.device(dev) = (dout.broadcast(size) * d_dx);
          d_ddy_flag = true;
        }
      }

      if (in_tensor_d_dy) {
        if (out_tensor_d_ddx) {
          ctx.template Alloc<T>(out_tensor_d_ddx);
          auto dout = EigenVector<T>::Flatten(*in_tensor_dout);
          auto d_dy = EigenVector<T>::Flatten(*in_tensor_d_dy);
          auto d_ddx = EigenVector<T>::Flatten(*out_tensor_d_ddx);
          Eigen::DSizes<int, 1> size(in_tensor_y->numel());
          d_ddx.device(dev) = (dout.broadcast(size) * d_dy);
          d_ddx_flag = true;
        }
      }

      if (in_tensor_d_ddout) {
        if (out_tensor_d_ddx) {
          ctx.template Alloc<T>(out_tensor_d_ddx);
          auto y = EigenVector<T>::Flatten(*in_tensor_y);
          auto d_ddout = EigenVector<T>::Flatten(*in_tensor_d_ddout);
          Eigen::DSizes<int, 1> size(in_tensor_y->numel());
          auto d_ddx = EigenVector<T>::Flatten(*out_tensor_d_ddx);
          if (d_ddx_flag) {
            d_ddx.device(dev) += (y * d_ddout.broadcast(size));
          } else {
            d_ddx.device(dev) = (y * d_ddout.broadcast(size));
          }
        }
        if (out_tensor_d_ddy) {
          ctx.template Alloc<T>(out_tensor_d_ddy);
          auto x = EigenVector<T>::Flatten(*in_tensor_x);
          auto d_ddout = EigenVector<T>::Flatten(*in_tensor_d_ddout);
          Eigen::DSizes<int, 1> size(in_tensor_x->numel());
          auto d_ddy = EigenVector<T>::Flatten(*out_tensor_d_ddy);
          if (d_ddy_flag) {
            d_ddy.device(dev) += (x * d_ddout.broadcast(size));
          } else {
            d_ddy.device(dev) = (x * d_ddout.broadcast(size));
          }
        }
      }
      if (out_tensor_d_x && !out_tensor_d_x->IsInitialized()) {
        FullLikeKernel<T, DeviceContext>(ctx,
                                         *in_tensor_x,
                                         Scalar(0.0),
                                         in_tensor_x->dtype(),
                                         out_tensor_d_x);
      }
      if (out_tensor_d_y && !out_tensor_d_y->IsInitialized()) {
        FullLikeKernel<T, DeviceContext>(ctx,
                                         *in_tensor_y,
                                         Scalar(0.0),
                                         in_tensor_y->dtype(),
                                         out_tensor_d_y);
      }
      if (out_tensor_d_dout && !out_tensor_d_dout->IsInitialized()) {
        FullLikeKernel<T, DeviceContext>(ctx,
                                         *in_tensor_dout,
                                         Scalar(0.0),
                                         in_tensor_dout->dtype(),
                                         out_tensor_d_dout);
      }
      if (out_tensor_d_ddx && !out_tensor_d_ddx->IsInitialized()) {
        FullLikeKernel<T, DeviceContext>(ctx,
                                         *in_tensor_x,
                                         Scalar(0.0),
                                         in_tensor_x->dtype(),
                                         out_tensor_d_ddx);
      }
      if (out_tensor_d_ddy && !out_tensor_d_ddy->IsInitialized()) {
        FullLikeKernel<T, DeviceContext>(ctx,
                                         *in_tensor_y,
                                         Scalar(0.0),
                                         in_tensor_y->dtype(),
                                         out_tensor_d_ddy);
      }
    }
#else
    const T* data_x = in_tensor_x->data<T>();
    const T* data_y = in_tensor_y->data<T>();
    const T* data_dout = in_tensor_dout->data<T>();
    const T* data_ddx = in_tensor_ddx ? in_tensor_ddx->data<T>() : nullptr;
    const T* data_ddy = in_tensor_ddy ? in_tensor_ddy->data<T>() : nullptr;
    const T* data_d_dx = in_tensor_d_dx ? in_tensor_d_dx->data<T>() : nullptr;
    const T* data_d_dy = in_tensor_d_dy ? in_tensor_d_dy->data<T>() : nullptr;
    const T* data_d_ddout =
        in_tensor_d_ddout ? in_tensor_d_ddout->data<T>() : nullptr;

    bool d_dout_flag = false;
    bool d_ddx_flag = false;
    bool d_ddy_flag = false;

    if (data_ddx) {
      if (out_tensor_d_y && data_d_ddout) {
        auto* data_d_y = ctx.template Alloc<T>(out_tensor_d_y);
        const DDim& dim = out_tensor_d_y->dims();
        size_t N = static_cast<size_t>(product(dim));
        auto _step = dim.size() > 0 ? dim[dim.size() - 1] : 1;
        auto step = _step != 0 ? _step : 1;
        int s = -1;
        for (size_t i = 0; i < N; ++i) {
          if (0 == i % step) ++s;
          data_d_y[i] = data_ddx[i] * data_d_ddout[s];
        }
      }
      if (out_tensor_d_dout && data_d_dy) {
        auto* data_d_dout = ctx.template Alloc<T>(out_tensor_d_dout);
        const DDim& dim = in_tensor_x->dims();
        size_t N = static_cast<size_t>(product(dim));
        auto _step = dim.size() > 0 ? dim[dim.size() - 1] : 1;
        auto step = _step != 0 ? _step : 1;
        int s = -1;
        bool new_s = false;
        for (size_t i = 0; i < N; ++i) {
          if (0 == i % step) {
            ++s;
            new_s = true;
          }
          if (new_s) {
            data_d_dout[s] = data_ddx[i] * data_d_dy[i];
          } else {
            data_d_dout[s] += data_ddx[i] * data_d_dy[i];
          }
          new_s = false;
        }
        d_dout_flag = true;
      }
    }

    if (data_ddy) {
      if (out_tensor_d_x && data_d_ddout) {
        auto* data_d_x = ctx.template Alloc<T>(out_tensor_d_x);
        const DDim& dim = out_tensor_d_x->dims();
        size_t N = static_cast<size_t>(product(dim));
        auto _step = dim.size() > 0 ? dim[dim.size() - 1] : 1;
        auto step = _step != 0 ? _step : 1;
        int s = -1;
        for (size_t i = 0; i < N; ++i) {
          if (0 == i % step) ++s;
          data_d_x[i] = data_ddy[i] * data_d_ddout[s];
        }
      }
      if (out_tensor_d_dout && data_d_dx) {
        auto* data_d_dout = ctx.template Alloc<T>(out_tensor_d_dout);
        const DDim& dim = in_tensor_x->dims();
        size_t N = static_cast<size_t>(product(dim));
        auto _step = dim.size() > 0 ? dim[dim.size() - 1] : 1;
        auto step = _step != 0 ? _step : 1;
        int s = -1;
        bool new_s = false;
        if (d_dout_flag) {
          for (size_t i = 0; i < N; ++i) {
            if (0 == i % step) {
              ++s;
            }
            data_d_dout[s] += data_ddy[i] * data_d_dx[i];
          }
        } else {
          for (size_t i = 0; i < N; ++i) {
            if (0 == i % step) {
              ++s;
              new_s = true;
            }
            if (new_s) {
              data_d_dout[s] = data_ddy[i] * data_d_dx[i];
            } else {
              data_d_dout[s] += data_ddy[i] * data_d_dx[i];
            }
            new_s = false;
          }
        }
      }
    }

    if (data_d_dx) {
      if (out_tensor_d_ddy) {
        auto* data_d_ddy = ctx.template Alloc<T>(out_tensor_d_ddy);
        const DDim& dim = out_tensor_d_ddy->dims();
        size_t N = static_cast<size_t>(product(dim));
        auto _step = dim.size() > 0 ? dim[dim.size() - 1] : 1;
        auto step = _step != 0 ? _step : 1;
        int s = -1;
        for (size_t i = 0; i < N; ++i) {
          if (0 == i % step) ++s;
          data_d_ddy[i] = data_dout[s] * data_d_dx[i];
        }
        d_ddy_flag = true;
      }
    }

    if (data_d_dy) {
      if (out_tensor_d_ddx) {
        auto* data_d_ddx = ctx.template Alloc<T>(out_tensor_d_ddx);
        const DDim& dim = out_tensor_d_ddx->dims();
        size_t N = static_cast<size_t>(product(dim));
        auto _step = dim.size() > 0 ? dim[dim.size() - 1] : 1;
        auto step = _step != 0 ? _step : 1;
        int s = -1;
        for (size_t i = 0; i < N; ++i) {
          if (0 == i % step) ++s;
          data_d_ddx[i] = data_dout[s] * data_d_dy[i];
        }
      }
      d_ddx_flag = true;
    }

    if (data_d_ddout) {
      if (out_tensor_d_ddx) {
        auto* data_d_ddx = ctx.template Alloc<T>(out_tensor_d_ddx);
        const DDim& dim = out_tensor_d_ddx->dims();
        size_t N = static_cast<size_t>(product(dim));
        auto _step = dim.size() > 0 ? dim[dim.size() - 1] : 1;
        auto step = _step != 0 ? _step : 1;
        int s = -1;
        if (d_ddx_flag) {
          for (size_t i = 0; i < N; ++i) {
            if (0 == i % step) ++s;
            data_d_ddx[i] += data_y[i] * data_d_ddout[s];
          }
        } else {
          for (size_t i = 0; i < N; ++i) {
            if (0 == i % step) ++s;
            data_d_ddx[i] = data_y[i] * data_d_ddout[s];
          }
        }
      }
      if (out_tensor_d_ddy) {
        auto* data_d_ddy = ctx.template Alloc<T>(out_tensor_d_ddy);
        const DDim& dim = out_tensor_d_ddy->dims();
        size_t N = static_cast<size_t>(product(dim));
        auto _step = dim.size() > 0 ? dim[dim.size() - 1] : 1;
        auto step = _step != 0 ? _step : 1;
        int s = -1;
        if (d_ddy_flag) {
          for (size_t i = 0; i < N; ++i) {
            if (0 == i % step) ++s;
            data_d_ddy[i] += data_x[i] * data_d_ddout[s];
          }
        } else {
          for (size_t i = 0; i < N; ++i) {
            if (0 == i % step) ++s;
            data_d_ddy[i] = data_x[i] * data_d_ddout[s];
          }
        }
      }
    }

    if (out_tensor_d_x && !out_tensor_d_x->IsInitialized()) {
      FullLikeKernel<T, DeviceContext>(
          ctx, *in_tensor_x, Scalar(0.0), in_tensor_x->dtype(), out_tensor_d_x);
    }
    if (out_tensor_d_y && !out_tensor_d_y->IsInitialized()) {
      FullLikeKernel<T, DeviceContext>(
          ctx, *in_tensor_y, Scalar(0.0), in_tensor_y->dtype(), out_tensor_d_y);
    }
    if (out_tensor_d_dout && !out_tensor_d_dout->IsInitialized()) {
      FullLikeKernel<T, DeviceContext>(ctx,
                                       *in_tensor_dout,
                                       Scalar(0.0),
                                       in_tensor_dout->dtype(),
                                       out_tensor_d_dout);
    }
    if (out_tensor_d_ddx && !out_tensor_d_ddx->IsInitialized()) {
      FullLikeKernel<T, DeviceContext>(ctx,
                                       *in_tensor_x,
                                       Scalar(0.0),
                                       in_tensor_x->dtype(),
                                       out_tensor_d_ddx);
    }
    if (out_tensor_d_ddy && !out_tensor_d_ddy->IsInitialized()) {
      FullLikeKernel<T, DeviceContext>(ctx,
                                       *in_tensor_y,
                                       Scalar(0.0),
                                       in_tensor_y->dtype(),
                                       out_tensor_d_ddy);
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
                         const DenseTensor& dout,
                         const paddle::optional<DenseTensor>& ddx,
                         const paddle::optional<DenseTensor>& ddy,
                         DenseTensor* dx,
                         DenseTensor* dy,
                         DenseTensor* ddout) {
  DotDoubleGradFunction<Context, T>()(
      dev_ctx, &x, &y, &dout, ddx.get_ptr(), ddy.get_ptr(), dx, dy, ddout);
}

template <typename T, typename Context>
void DotTripleGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& y,
                         const DenseTensor& dout,
                         const paddle::optional<DenseTensor>& ddx,
                         const paddle::optional<DenseTensor>& ddy,
                         const paddle::optional<DenseTensor>& d_dx,
                         const paddle::optional<DenseTensor>& d_dy,
                         const paddle::optional<DenseTensor>& d_ddout,
                         DenseTensor* d_x,
                         DenseTensor* d_y,
                         DenseTensor* d_ddx,
                         DenseTensor* d_ddy,
                         DenseTensor* d_dout) {
  DotTripleGradFunction<Context, T>()(dev_ctx,
                                      &x,
                                      &y,
                                      &dout,
                                      ddx.get_ptr(),
                                      ddy.get_ptr(),
                                      d_dx.get_ptr(),
                                      d_dy.get_ptr(),
                                      d_ddout.get_ptr(),
                                      d_x,
                                      d_y,
                                      d_dout,
                                      d_ddx,
                                      d_ddy);
}

}  // namespace phi
