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
#include "paddle/phi/kernels/funcs/abs_util.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
// #include "paddle/phi/kernels/funcs/element_util.h"
namespace phi {

template <typename T, typename Enable = void>
struct CudaAbsFunctor;

template <typename T>
struct CudaAbsFunctor<T, phi::funcs::Complex<T, phi::dtype::Real<T>>> {
  __device__ __forceinline__ phi::dtype::Real<T> operator()(const T* x) const {
    return abs(x[0]);
  }
};

template <typename T>
struct CudaAbsFunctor<
    T,
    std::enable_if_t<std::is_same<T, phi::dtype::Real<T>>::value &&
                     std::is_same<T, phi::dtype::bfloat16>::value>> {
  __device__ __forceinline__ T operator()(const T* x) const {
    return abs(x[0]);
  }
};

template <typename T>
struct CudaAbsFunctor<
    T,
    std::enable_if_t<std::is_same<T, phi::dtype::Real<T>>::value &&
                     !std::is_same<T, phi::dtype::bfloat16>::value>> {
  __device__ __forceinline__ T operator()(const T* x) const {
    return std::abs(x[0]);
  }
};

// template <typename T, typename Enable = void>
// struct CudaAbsFunctor;

// template <typename T>
// struct CudaAbsFunctor<T, phi::funcs::Complex<T, phi::dtype::Real<T>>> {
//   __device__ __forceinline__ phi::dtype::Real<T> operator()(const T x) const
//   {
//     return abs(x);
//   }
// };

// template <typename T>
// struct CudaAbsFunctor<
//     T,
//     std::enable_if_t<std::is_same<T, phi::dtype::Real<T>>::value &&
//                      std::is_same<T, phi::dtype::bfloat16>::value>> {
//   __device__ __forceinline__ T operator()(const T x) const { return abs(x); }
// };

// template <typename T>
// struct CudaAbsFunctor<
//     T,
//     std::enable_if_t<std::is_same<T, phi::dtype::Real<T>>::value &&
//                      !std::is_same<T, phi::dtype::bfloat16>::value>> {
//   __device__ __forceinline__ T operator()(const T x) const {
//     return std::abs(x);
//   }
// };

template <typename T, typename Context>
void AbsKernel(const Context& ctx, const DenseTensor& x, DenseTensor* out) {
  ctx.template Alloc<phi::dtype::Real<T>>(out);
  std::vector<const DenseTensor*> ins = {&x};
  std::vector<DenseTensor*> outs = {out};
  // auto functor = CudaAbsFunctor<T>();

  // funcs::ElementwiseKernel<phi::dtype::Real<T>>(ctx, ins, &outs, functor);

  // auto t = funcs::UnaryFunctor<T, CudaAbsFunctor<T> >( CudaAbsFunctor<T>() );
  // funcs::abs<T>(x, out, t);

  // auto functor = CudaAbsFunctor<T>();
  // //auto t = funcs::AnyFunctor<T, CudaAbsFunctor<T> >( CudaAbsFunctor<T>() );
  // funcs::LaunchSameDimsElementwiseCudaKernel<funcs::ElementwiseType::kUnary,
  // T,
  //                                       phi::dtype::Real<T>>(ctx, ins, &outs,
  //                                                      functor);

  // funcs::test_func( []( T a) { std::cout << "111 " << a << std::endl;} );

  // auto t = funcs::UnaryFunctor<T, phi::dtype::Real<T>, CudaAbsFunctor<T> >(
  // CudaAbsFunctor<T>() );
  std::cerr << "out dim " << out->dims() << std::endl;
  auto t = CudaAbsFunctor<T>();
  funcs::TensorContainer ten_con(&x, out);
  std::cerr << "out ptr" << out->data() << std::endl;
  std::cerr << "out ptr 12 " << ten_con.out_->data() << std::endl;
  funcs::test_func(ctx.stream(), ten_con, t);
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
