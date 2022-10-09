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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/impl/compare_kernel_impl.h"

#ifdef PADDLE_WITH_XPU_KP
#include "paddle/phi/backends/xpu/xpu_context.h"
#else
#include <thrust/fill.h>

#include <vector>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/compare_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/gpu/reduce.h"
#include "paddle/phi/kernels/primitive/functor_primitives.h"
#endif

namespace phi {

template <typename T>
struct BitwiseAdd {
  // Bitwise add operator, returns <tt>a + b</tt>
  inline T initial() { return static_cast<T>(true); }

  __host__ __device__ __forceinline__ T operator()(const T& a,
                                                   const T& b) const {
    return a & b;
  }
};

template <typename T,
          typename Context,
          typename Functor,
          typename InverseFunctor>
inline void CompareKernelImpl(const Context& ctx,
                              const DenseTensor& x,
                              const DenseTensor& y,
                              int axis,
                              DenseTensor* out) {
  ctx.template Alloc<bool>(out);
  std::vector<const DenseTensor*> ins{&x, &y};
  std::vector<DenseTensor*> outs{out};
  funcs::BroadcastKernel<ElementwiseType::kBinary, T, bool>(
      ctx, ins, &outs, axis, Functor());
}

#ifndef PADDLE_WITH_XPU_KP
template <typename T, typename Context, typename Functor>
inline void CompareAllKernelImpl(const Context& ctx,
                                 const DenseTensor& x,
                                 const DenseTensor& y,
                                 DenseTensor* out) {
  bool* out_data = ctx.template Alloc<bool>(out);

  if (x.dims() != y.dims()) {
    thrust::device_ptr<bool> out_dev_ptr(out_data);
    thrust::fill(out_dev_ptr, out_dev_ptr + 1, false);
    return;
  }

  DenseTensor tmp;
  tmp.Resize(x.dims());
  ctx.template Alloc<bool>(&tmp);

  std::vector<const DenseTensor*> ins{&x, &y};
  std::vector<DenseTensor*> outs{&tmp};
  funcs::ElementwiseKernel<bool>(ctx, ins, &outs, Functor());

  // Reduce by 'bitwise and' operator
  std::vector<int> reduce_dims;
  reduce_dims.resize(tmp.dims().size());
  for (int i = 0; i < reduce_dims.size(); ++i) {
    reduce_dims[i] = i;
  }
  funcs::ReduceKernel<bool, bool, BitwiseAdd, kps::IdentityFunctor<bool>>(
      ctx, tmp, out, kps::IdentityFunctor<bool>(), reduce_dims);
}
#endif

}  // namespace phi

#ifdef PADDLE_WITH_XPU_KP
PD_REGISTER_KERNEL(less_than, KPS, ALL_LAYOUT, phi::LessThanKernel, int) {}
PD_REGISTER_KERNEL(less_equal, KPS, ALL_LAYOUT, phi::LessEqualKernel, int) {}
PD_REGISTER_KERNEL(greater_than, KPS, ALL_LAYOUT, phi::GreaterThanKernel, int) {
}
PD_REGISTER_KERNEL(
    greater_equal, KPS, ALL_LAYOUT, phi::GreaterEqualKernel, int) {}
PD_REGISTER_KERNEL(equal, KPS, ALL_LAYOUT, phi::EqualKernel, int) {}
PD_REGISTER_KERNEL(not_equal, KPS, ALL_LAYOUT, phi::NotEqualKernel, int) {}
#else
PD_REGISTER_KERNEL(less_than,
                   KPS,
                   ALL_LAYOUT,
                   phi::LessThanKernel,
                   bool,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(less_equal,
                   KPS,
                   ALL_LAYOUT,
                   phi::LessEqualKernel,
                   bool,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(greater_than,
                   KPS,
                   ALL_LAYOUT,
                   phi::GreaterThanKernel,
                   bool,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(greater_equal,
                   KPS,
                   ALL_LAYOUT,
                   phi::GreaterEqualKernel,
                   bool,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(equal,
                   KPS,
                   ALL_LAYOUT,
                   phi::EqualKernel,
                   bool,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(not_equal,
                   KPS,
                   ALL_LAYOUT,
                   phi::NotEqualKernel,
                   bool,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(equal_all,
                   KPS,
                   ALL_LAYOUT,
                   phi::EqualAllKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double) {}
#endif
