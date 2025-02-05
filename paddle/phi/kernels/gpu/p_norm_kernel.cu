// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/p_norm_kernel.h"

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/funcs/p_norm_utils.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/gpu/reduce.h"

namespace phi {
template <typename T>
struct NonzeroFunctor {
  HOSTDEVICE explicit inline NonzeroFunctor() = default;
  HOSTDEVICE inline T operator()(const T x) const {
    return static_cast<T>(static_cast<double>(x) != 0);
  }
};

template <typename T>
struct AbsFunctor {
  HOSTDEVICE explicit inline AbsFunctor() = default;
  HOSTDEVICE inline T operator()(const T x) const {
    return static_cast<T>(inline_abs(x));
  }
};

template <typename T>
struct UnsignedPowFunctor {
  HOSTDEVICE explicit inline UnsignedPowFunctor(float porder) {
    this->porder = porder;
  }
  HOSTDEVICE inline T operator()(const T x) const {
    return static_cast<T>(inline_pow(inline_abs(x), static_cast<T>(porder)));
  }
  float porder;
};

#ifndef _WIN32
// To avoid large .so size in Windows cuda11.8
template <typename T>
struct FabsFunctor {
  HOSTDEVICE explicit inline FabsFunctor() = default;
  HOSTDEVICE inline T operator()(const T x) const {
    return static_cast<T>(inline_fabs(x));
  }
};

template <typename T>
struct SquareFunctor {
  HOSTDEVICE explicit inline SquareFunctor() = default;
  HOSTDEVICE inline T operator()(const T x) const {
    return static_cast<T>(inline_square(x));
  }
};

template <typename T>
struct FabsCubicFunctor {
  HOSTDEVICE explicit inline FabsCubicFunctor() = default;
  HOSTDEVICE inline T operator()(const T x) const {
    return static_cast<T>(inline_fabs_cubic(x));
  }
};
#endif

#ifndef PADDLE_WITH_XPU_KP

inline void GetDims(const phi::DDim& dim,
                    int axis,
                    int* pre,
                    int* n,
                    int* post,
                    bool asvector) {
  *pre = 1;
  *post = 1;
  *n = static_cast<int>(dim[axis]);
  if (asvector) {
    *n = static_cast<int>(product(dim));
  } else {
    for (int i = 0; i < axis; ++i) {
      (*pre) *= static_cast<int>(dim[i]);
    }
    for (int i = axis + 1; i < dim.size(); ++i) {
      (*post) *= static_cast<int>(dim[i]);
    }
  }
}

template <typename T, typename Context>
void ReducePNormEigen(const Context& dev_ctx,
                      const DenseTensor& x,
                      float porder,
                      int axis,
                      float epsilon,
                      bool keepdim,
                      bool asvector,
                      DenseTensor* out) {
  auto xdim = x.dims();
  if (axis < 0) axis = xdim.size() + axis;
  int pre = 0, n = 0, post = 0;
  GetDims(xdim, axis, &pre, &n, &post, asvector);

  for (int i = 0; i < xdim.size(); i++) {
    PADDLE_ENFORCE_LT(0,
                      xdim[i],
                      errors::InvalidArgument(
                          "The dims of Input(X) should be greater than 0."));
  }

  auto* place = dev_ctx.eigen_device();

  Eigen::DSizes<int, 3> shape(pre, n, post);
  Eigen::DSizes<int, 2> norm_shape(pre, post);

  auto x_e = phi::EigenVector<T>::Flatten(x);
  auto norm_e = phi::EigenVector<T>::Flatten(*out);

  auto xr = x_e.reshape(shape);
  auto norm = norm_e.reshape(norm_shape);

  // p=0 means number of non-zero elements of (xr)
  // p=inf means the maximum of |xr|
  // p=-inf means the minimum of |xr|
  // otherwise, Lp-norm = pow(sum(pow(|xr|, p)), 1/p)
  Eigen::DSizes<int, 1> rdim(1);
  if (porder == 0) {
    norm.device(*place) =
        (xr != xr.constant(static_cast<T>(0))).template cast<T>().sum(rdim);
  } else if (porder == INFINITY) {
    norm.device(*place) = xr.abs().maximum(rdim);
  } else if (porder == -INFINITY) {
    norm.device(*place) = xr.abs().minimum(rdim);
  } else {
    norm.device(*place) = xr.abs()
                              .pow(static_cast<T>(porder))
                              .sum(rdim)
                              .pow(static_cast<T>(1.0f / porder));
  }
}
#endif

template <typename T, typename Context>
void PNormKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 float porder,
                 int axis,
                 float epsilon,
                 bool keepdim,
                 bool asvector,
                 DenseTensor* out) {
  auto* in_x = &x;
  auto* out_norm = out;
  T* norm = dev_ctx.template Alloc<T>(out);
  auto xdim = in_x->dims();
  std::vector<int64_t> axis_dims = {static_cast<int64_t>(axis)};
  std::vector<int> reduce_axis =
      funcs::details::GetReduceDim(axis_dims, xdim.size(), asvector);

  for (int i = 0; i < xdim.size(); i++) {
    PADDLE_ENFORCE_LT(0,
                      xdim[i],
                      errors::InvalidArgument(
                          "The dims of Input(X) should be greater than 0."));
  }

  if (x.numel() > std::numeric_limits<int32_t>::max()) {
#ifndef PADDLE_WITH_XPU_KP
    ReducePNormEigen<T, Context>(
        dev_ctx, *in_x, porder, axis, epsilon, keepdim, asvector, out_norm);
#else
    PADDLE_THROW(common::errors::Fatal(
        "If Input.numel() > INT32_MAX, reduce_sum kernel uses EigenTensor "
        "sum for reduce_sum function. Such case is only supported on GPU "
        "now."));
#endif
  } else {
    using MT = typename dtype::MPTypeTrait<T>::Type;
    if (porder == 0) {
      phi::funcs::ReduceKernel<T, T, kps::AddFunctor, NonzeroFunctor<T>>(
          dev_ctx, *in_x, out_norm, NonzeroFunctor<T>(), reduce_axis);
    } else if (porder == INFINITY) {
      phi::funcs::ReduceKernel<T, T, kps::MaxFunctor, AbsFunctor<T>>(
          dev_ctx, *in_x, out_norm, AbsFunctor<T>(), reduce_axis);
    } else if (porder == -INFINITY) {
      phi::funcs::ReduceKernel<T, T, kps::MinFunctor, AbsFunctor<T>>(
          dev_ctx, *in_x, out_norm, AbsFunctor<T>(), reduce_axis);
    } else {
#ifdef _WIN32
      phi::funcs::ReduceKernel<T, T, kps::AddFunctor, UnsignedPowFunctor<T>>(
          dev_ctx, *in_x, out_norm, UnsignedPowFunctor<T>(porder), reduce_axis);

      const DenseTensor* tmp_norm = out_norm;
      std::vector<const DenseTensor*> ins = {tmp_norm};
      std::vector<DenseTensor*> outs = {out_norm};
      phi::funcs::ElementwiseKernel<T>(
          dev_ctx, ins, &outs, UnsignedPowFunctor<T>(1. / porder));
#else
      if (porder == 1.0) {
        // fast 1-norm
        phi::funcs::ReduceKernel<T, T, kps::AddFunctor, FabsFunctor<T>>(
            dev_ctx, *in_x, out_norm, FabsFunctor<T>(), reduce_axis);
      } else if (porder == 2.0) {
        // fast 2-norm
        phi::funcs::ReduceKernel<T, T, kps::AddFunctor, SquareFunctor<T>>(
            dev_ctx, *in_x, out_norm, SquareFunctor<T>(), reduce_axis);
      } else if (porder == 3.0) {
        // fast 3-norm
        phi::funcs::ReduceKernel<T, T, kps::AddFunctor, FabsCubicFunctor<T>>(
            dev_ctx, *in_x, out_norm, FabsCubicFunctor<T>(), reduce_axis);
      } else {
        // vanilla norm
        phi::funcs::ReduceKernel<T, T, kps::AddFunctor, UnsignedPowFunctor<T>>(
            dev_ctx,
            *in_x,
            out_norm,
            UnsignedPowFunctor<T>(porder),
            reduce_axis);
      }

      if (porder != 1.0) {
        // save computation when porder is 1.0
        const DenseTensor* tmp_norm = out_norm;
        std::vector<const DenseTensor*> ins = {tmp_norm};
        std::vector<DenseTensor*> outs = {out_norm};
        phi::funcs::ElementwiseKernel<T>(
            dev_ctx, ins, &outs, UnsignedPowFunctor<T>(1. / porder));
      }
#endif
    }
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(p_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::PNormKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
