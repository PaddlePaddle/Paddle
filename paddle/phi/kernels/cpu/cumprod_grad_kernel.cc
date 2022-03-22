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

#include "paddle/phi/kernels/cumprod_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/cumprod.h"
#include "paddle/phi/kernels/funcs/for_range.h"

// NOTE(@xiongkun): use of IsComplex<>
#include "paddle/fluid/framework/data_type.h"

namespace phi {
template <typename T, typename Context>
void CumprodGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& out,
                       const DenseTensor& d_out,
                       int dim,
                       DenseTensor* d_x) {
  DDim shape = x.dims();

  auto* d_out_data = d_out.data<T>();
  auto* x_data = x.data<T>();
  auto* out_data = out.data<T>();
  auto* d_x_data = dev_ctx.template Alloc<T>(d_x);

  size_t outer_dim = 1;
  size_t mid_dim = 1;
  size_t inner_dim = 1;
  GetCumprodDimInfo(shape, dim, &outer_dim, &mid_dim, &inner_dim);
  size_t numel = outer_dim * mid_dim * inner_dim;

  // deal with complex
  const T* x_data_deal;
  const T* out_data_deal;
  Allocator::AllocationPtr x_conj;
  Allocator::AllocationPtr out_conj;
  if (paddle::framework::IsComplex<T>::value) {
    x_conj = const_cast<Allocator&>(dev_ctx.GetAllocator())
                 .Allocate(numel * sizeof(T));
    auto* x_data_conj = reinterpret_cast<T*>(x_conj->ptr());
    out_conj = const_cast<Allocator&>(dev_ctx.GetAllocator())
                   .Allocate(numel * sizeof(T));
    auto* out_data_conj = reinterpret_cast<T*>(out_conj->ptr());

    phi::funcs::ForRange<Context> for_range_x(dev_ctx, numel);
    phi::funcs::ConjFunctor<T> functor_x(x_data, numel, x_data_conj);
    for_range_x(functor_x);

    phi::funcs::ForRange<Context> for_range_out(dev_ctx, numel);
    phi::funcs::ConjFunctor<T> functor_out(out_data, numel, out_data_conj);
    for_range_out(functor_out);

    x_data_deal = x_data_conj;
    out_data_deal = out_data_conj;
  } else {
    x_data_deal = x_data;
    out_data_deal = out_data;
  }

  for (size_t i = 0; i < outer_dim; i++) {
    for (size_t k = 0; k < inner_dim; k++) {
      for (size_t j = 0; j < mid_dim; j++) {
        size_t index = i * mid_dim * inner_dim + j * inner_dim + k;
        d_x_data[index] = 0;
        for (size_t n = 0; n < mid_dim; n++) {
          size_t pos = i * mid_dim * inner_dim + n * inner_dim + k;
          T elem;
          if (j == 0) {
            elem = d_out_data[pos];
          } else {
            elem = d_out_data[pos] * out_data_deal[index - inner_dim];
          }
          if (pos > index) {
            for (size_t m = index + inner_dim; m <= pos; m += inner_dim) {
              elem *= x_data_deal[m];
            }
          } else if (pos < index) {
            elem = static_cast<T>(0);
          }
          d_x_data[index] += elem;
        }
      }
    }
  }
}
}  // namespace phi
PD_REGISTER_KERNEL(cumprod_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::CumprodGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
