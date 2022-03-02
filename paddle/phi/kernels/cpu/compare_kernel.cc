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
#include "paddle/phi/kernels/impl/compare_kernel_impl.h"

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"

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
        ctx, x, y, axis, Functor(), out);
  } else {
    funcs::ElementwiseCompute<InverseFunctor, T, bool>(
        ctx, x, y, axis, InverseFunctor(), out);
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
          ctx, x, y, 0, Functor(), &tmp);
    }
    auto tmp_flat = EigenVector<bool>::Flatten(tmp);
    auto out_es = EigenScalar<bool>::From(*out);
    auto& place = *ctx.eigen_device();
    auto reduce_dim = Eigen::array<int, 1>({{0}});
    out_es.device(place) = tmp_flat.all(reduce_dim);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(less_than,
                   CPU,
                   ALL_LAYOUT,
                   phi::LessThanKernel,
                   bool,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double) {}
PD_REGISTER_KERNEL(less_equal,
                   CPU,
                   ALL_LAYOUT,
                   phi::LessEqualKernel,
                   bool,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double) {}
PD_REGISTER_KERNEL(greater_than,
                   CPU,
                   ALL_LAYOUT,
                   phi::GreaterThanKernel,
                   bool,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double) {}
PD_REGISTER_KERNEL(greater_equal,
                   CPU,
                   ALL_LAYOUT,
                   phi::GreaterEqualKernel,
                   bool,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double) {}
PD_REGISTER_KERNEL(equal,
                   CPU,
                   ALL_LAYOUT,
                   phi::EqualKernel,
                   bool,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double) {}
PD_REGISTER_KERNEL(not_equal,
                   CPU,
                   ALL_LAYOUT,
                   phi::NotEqualKernel,
                   bool,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double) {}

PD_REGISTER_KERNEL(equal_all,
                   CPU,
                   ALL_LAYOUT,
                   phi::EqualAllKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double) {}
