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

#include "paddle/phi/kernels/compare_kernel.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/impl/compare_kernel_impl.h"

namespace phi {

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
  if (x.dims().size() >= y.dims().size()) {
    funcs::ElementwiseCompute<Functor, T, bool>(
        ctx, x, y, Functor(), out, axis);
  } else {
    funcs::ElementwiseCompute<InverseFunctor, T, bool>(
        ctx, x, y, InverseFunctor(), out, axis);
  }
}

template <typename T,
          typename Context,
          typename Functor,
          typename InverseFunctor>
inline void InplaceCompareKernelImpl(const Context& ctx,
                                     const DenseTensor& x,
                                     const DenseTensor& y,
                                     int axis,
                                     DenseTensor* out) {
  auto x_origin = x;
  out->set_type(phi::DataType::BOOL);
  ctx.template Alloc<bool>(out);
  if (x_origin.dims().size() >= y.dims().size()) {
    funcs::ElementwiseCompute<Functor, T, bool>(
        ctx, x_origin, y, Functor(), out, axis);
  } else {
    funcs::ElementwiseCompute<InverseFunctor, T, bool>(
        ctx, x_origin, y, InverseFunctor(), out, axis);
  }
}

template <typename T, typename Context, typename Functor>
inline void CompareAllKernelImpl(const Context& ctx,
                                 const DenseTensor& x,
                                 const DenseTensor& y,
                                 DenseTensor* out) {
  bool* out_data = ctx.template Alloc<bool>(out);

  if (x.dims() != y.dims()) {
    out_data[0] = false;
  } else {
    DenseTensor tmp;
    tmp.Resize(x.dims());
    ctx.template Alloc<bool>(&tmp);

    if (x.numel() == 1 && y.numel() == 1) {
      bool* tmp_data = tmp.data<bool>();
      tmp_data[0] = Functor()(x.data<T>()[0], y.data<T>()[0]);
    } else {
      funcs::ElementwiseCompute<Functor, T, bool>(
          ctx, x, y, Functor(), &tmp, 0);
    }
    auto tmp_flat = EigenVector<bool>::Flatten(tmp);
    auto out_es = EigenScalar<bool>::From(*out);
    auto& place = *ctx.eigen_device();
    auto reduce_dim = Eigen::array<int, 1>({{0}});
    out_es.device(place) = tmp_flat.all(reduce_dim);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(equal_all,
                   CPU,
                   ALL_LAYOUT,
                   phi::EqualAllKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

#define PD_REGISTER_COMPLEX_COMPARE_KERNEL(name, func)    \
  PD_REGISTER_KERNEL(name,                                \
                     CPU,                                 \
                     ALL_LAYOUT,                          \
                     phi::func##Kernel,                   \
                     bool,                                \
                     int,                                 \
                     uint8_t,                             \
                     int8_t,                              \
                     int16_t,                             \
                     int64_t,                             \
                     phi::dtype::complex<float>,          \
                     phi::dtype::complex<double>,         \
                     float,                               \
                     double,                              \
                     phi::dtype::float16,                 \
                     phi::dtype::bfloat16) {              \
    kernel->OutputAt(0).SetDataType(phi::DataType::BOOL); \
  }

#define PD_REGISTER_COMPARE_KERNEL(name, func)            \
  PD_REGISTER_KERNEL(name,                                \
                     CPU,                                 \
                     ALL_LAYOUT,                          \
                     phi::func##Kernel,                   \
                     bool,                                \
                     int,                                 \
                     uint8_t,                             \
                     int8_t,                              \
                     int16_t,                             \
                     int64_t,                             \
                     float,                               \
                     double,                              \
                     phi::dtype::float16,                 \
                     phi::dtype::bfloat16) {              \
    kernel->OutputAt(0).SetDataType(phi::DataType::BOOL); \
  }

PD_REGISTER_COMPARE_KERNEL(less_than, LessThan)
PD_REGISTER_COMPARE_KERNEL(less_equal, LessEqual)
PD_REGISTER_COMPARE_KERNEL(greater_than, GreaterThan)
PD_REGISTER_COMPARE_KERNEL(greater_equal, GreaterEqual)

PD_REGISTER_COMPLEX_COMPARE_KERNEL(equal, Equal)
PD_REGISTER_COMPLEX_COMPARE_KERNEL(not_equal, NotEqual)
