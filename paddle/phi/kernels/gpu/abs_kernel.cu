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

#include "paddle/phi/kernels/abs_kernel.h"

#include <algorithm>
#include <vector>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/gpu/unary.h"

namespace phi {

template <typename T, typename Enable = void>
struct CudaAbsFunctor;

template <typename T>
struct CudaAbsFunctor<T, phi::funcs::Complex<T, phi::dtype::Real<T>>> {
  __device__ __forceinline__ phi::dtype::Real<T> operator()(const T x) const {
    return abs(x);
  }
};

template <typename T>
struct CudaAbsFunctor<
    T,
    std::enable_if_t<std::is_same<T, phi::dtype::Real<T>>::value &&
                     std::is_same<T, phi::dtype::bfloat16>::value>> {
  __device__ __forceinline__ T operator()(const T x) const { return abs(x); }
};

template <typename T>
struct CudaAbsFunctor<
    T,
    std::enable_if_t<std::is_same<T, phi::dtype::Real<T>>::value &&
                     !std::is_same<T, phi::dtype::bfloat16>::value>> {
  __device__ __forceinline__ T operator()(const T x) const {
    return std::abs(x);
  }
};

template <typename T, typename Context>
void AbsKernel(const Context& ctx, const DenseTensor& x, DenseTensor* out) {
  ctx.template Alloc<phi::dtype::Real<T>>(out);
  auto functor = CudaAbsFunctor<T>();
  UnaryKernel<phi::dtype::Real<T>, Context>(ctx, &x, out, functor);
}

}  // namespace phi

PD_REGISTER_KERNEL(abs,
                   GPU,
                   ALL_LAYOUT,
                   phi::AbsKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}
