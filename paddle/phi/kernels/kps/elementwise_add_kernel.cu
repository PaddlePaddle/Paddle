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

#include "paddle/phi/kernels/elementwise_add_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#ifndef PADDLE_WITH_XPU_KP
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#endif
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"

namespace phi {

template <typename T, typename Context>
void AddKernelImpl(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   int axis,
                   DenseTensor* out) {
  std::vector<const DenseTensor*> inputs = {&x, &y};
  std::vector<DenseTensor*> outputs = {out};
  dev_ctx.template Alloc<T>(out);
  funcs::BroadcastKernel<T>(
      dev_ctx, inputs, &outputs, funcs::AddFunctor<T>(), axis);
}

template <typename T, typename Context>
void MultiPrecisionAddKernelImpl(const Context& dev_ctx,
                                 const DenseTensor& x,
                                 const DenseTensor& y,
                                 DenseTensor* out) {
  std::vector<const DenseTensor*> inputs = {&x, &y};
  std::vector<DenseTensor*> outputs = {out};
  dev_ctx.template Alloc<T>(out);
  if (y.dtype() == phi::DataType::BFLOAT16) {
    funcs::ElementwiseKernel<T>(
        dev_ctx,
        inputs,
        &outputs,
        funcs::MultiPrecisionAddFunctor<T, phi::bfloat16>());
  } else if (y.dtype() == phi::DataType::FLOAT16) {
    funcs::ElementwiseKernel<T>(
        dev_ctx,
        inputs,
        &outputs,
        funcs::MultiPrecisionAddFunctor<T, phi::float16>());
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Unsupport x dtype:%s, y dtype:%s for add(x, y) operation",
        phi::DataTypeToString(x.type()),
        phi::DataTypeToString(y.type())));
  }
}

template <typename T, typename Context>
void AddKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const DenseTensor& y,
               DenseTensor* out) {
#ifdef PADDLE_WITH_CUDA
  if (x.dtype() == phi::DataType::FLOAT32 &&
      (y.dtype() == phi::DataType::BFLOAT16 ||
       y.dtype() == phi::DataType::FLOAT16)) {
    MultiPrecisionAddKernelImpl<float, Context>(dev_ctx, x, y, out);
  } else {
#endif
    AddKernelImpl<T, Context>(dev_ctx, x, y, -1, out);
#ifdef PADDLE_WITH_CUDA
  }
#endif
}

template <typename T, typename Context>
void GradAddKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* out) {
  AddKernelImpl<T>(dev_ctx, x, y, -1, out);
}

}  // namespace phi

#ifdef PADDLE_WITH_XPU_KP
PD_REGISTER_KERNEL(add, KPS, ALL_LAYOUT, phi::AddKernel, float) {}
#else

using float16 = phi::dtype::float16;
using bfloat16 = phi::dtype::bfloat16;
using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

PD_REGISTER_KERNEL(add,
                   KPS,
                   ALL_LAYOUT,
                   phi::AddKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   complex64,
                   complex128) {}

PD_REGISTER_KERNEL(grad_add,
                   KPS,
                   ALL_LAYOUT,
                   phi::GradAddKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   complex64,
                   complex128) {}
#endif
