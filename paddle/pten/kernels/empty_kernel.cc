/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License. */
#include "paddle/pten/kernels/empty_kernel.h"

#include "paddle/pten/backends/all_context.h"
#include "paddle/pten/core/kernel_registry.h"

#include "paddle/pten/common/complex.h"

namespace pten {

template <typename T, typename Context>
void EmptyKernel(const Context& dev_ctx,
                 const ScalarArray& shape,
                 DenseTensor* out) {
  out->ResizeAndAllocate(pten::make_ddim(shape.GetData()));
}

template <typename T, typename Context>
void EmptyLikeKernel(const Context& dev_ctx, DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
}

}  // namespace pten

PT_REGISTER_KERNEL(empty,
                   CPU,
                   ALL_LAYOUT,
                   pten::EmptyKernel,
                   float,
                   double,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   pten::dtype::float16,
                   pten::dtype::bfloat16,
                   pten::dtype::complex<float>,
                   pten::dtype::complex<double>) {}

PT_REGISTER_KERNEL(empty_like,
                   CPU,
                   ALL_LAYOUT,
                   pten::EmptyLikeKernel,
                   float,
                   double,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   pten::dtype::float16,
                   pten::dtype::bfloat16,
                   pten::dtype::complex<float>,
                   pten::dtype::complex<double>) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PT_REGISTER_KERNEL(empty,
                   GPU,
                   ALL_LAYOUT,
                   pten::EmptyKernel,
                   float,
                   double,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   pten::dtype::float16,
                   pten::dtype::complex<float>,
                   pten::dtype::complex<double>) {}

PT_REGISTER_KERNEL(empty_like,
                   GPU,
                   ALL_LAYOUT,
                   pten::EmptyLikeKernel,
                   float,
                   double,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   pten::dtype::float16,
                   pten::dtype::bfloat16,
                   pten::dtype::complex<float>,
                   pten::dtype::complex<double>) {}
#endif
