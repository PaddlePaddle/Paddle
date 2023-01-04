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
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/gpu/reduce.h"

namespace phi {

template <typename T>
__device__ __forceinline__ int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

__device__ __forceinline__ dtype::float16 inline_abs(dtype::float16 x) {
  return static_cast<dtype::float16>(abs(static_cast<float>(x)));
}

__device__ __forceinline__ dtype::bfloat16 inline_abs(dtype::bfloat16 x) {
  return static_cast<dtype::bfloat16>(abs(static_cast<float>(x)));
}

__device__ __forceinline__ float inline_abs(float x) { return abs(x); }
__device__ __forceinline__ double inline_abs(double x) { return abs(x); }

__device__ __forceinline__ int inline_sign(dtype::float16 x) {
  return sgn<dtype::float16>(x);
}
__device__ __forceinline__ int inline_sign(float x) { return sgn<float>(x); }
__device__ __forceinline__ int inline_sign(double x) { return sgn<double>(x); }

__device__ __forceinline__ dtype::float16 inline_pow(dtype::float16 base,
                                                     dtype::float16 exponent) {
  return static_cast<dtype::float16>(
      pow(static_cast<float>(base), static_cast<float>(exponent)));
}
__device__ __forceinline__ dtype::bfloat16 inline_pow(
    dtype::bfloat16 base, dtype::bfloat16 exponent) {
  return static_cast<dtype::bfloat16>(
      pow(static_cast<float>(base), static_cast<float>(exponent)));
}
__device__ __forceinline__ float inline_pow(float base, float exponent) {
  return pow(base, exponent);
}
__device__ __forceinline__ double inline_pow(double base, double exponent) {
  return pow(base, exponent);
}

template <typename T>
struct NonzeroFunctor {
  HOSTDEVICE explicit inline NonzeroFunctor() {}
  HOSTDEVICE inline T operator()(const T x) const {
    return static_cast<T>(static_cast<double>(x) != 0);
  }
};

template <typename T>
struct AbsFunctor {
  HOSTDEVICE explicit inline AbsFunctor() {}
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
    phi::funcs::ReduceKernel<T, T, kps::AddFunctor, UnsignedPowFunctor<T>>(
        dev_ctx, *in_x, out_norm, UnsignedPowFunctor<T>(porder), reduce_axis);

    const DenseTensor* tmp_norm = out_norm;
    std::vector<const DenseTensor*> ins = {tmp_norm};
    std::vector<DenseTensor*> outs = {out_norm};
    phi::funcs::ElementwiseKernel<T>(
        dev_ctx, ins, &outs, UnsignedPowFunctor<T>(1. / porder));
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
