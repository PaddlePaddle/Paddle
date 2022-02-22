//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/flatten_kernel.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

namespace phi {

template <typename T, typename Context>
void FlattenKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   int start_axis,
                   int stop_axis,
                   DenseTensor* out) {
  auto out_dims = out->dims();
  phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
  out->Resize(out_dims);
}

// TODO(yuanrisheng): this kernel is for training and xshape is a Intermediate
// Output Tensor，
// is there a more flexible way to deal with this case?
template <typename T, typename Context>
void FlattenWithXShape(const Context& dev_ctx,
                       const DenseTensor& x,
                       int start_axis,
                       int stop_axis,
                       DenseTensor* out,
                       DenseTensor* xshape) {
  FlattenKernel<T, Context>(dev_ctx, x, start_axis, stop_axis, out);
  funcs::SetXShape(x, xshape);
}

}  // namespace phi

PD_REGISTER_KERNEL(flatten,
                   CPU,
                   ALL_LAYOUT,
                   phi::FlattenKernel,
                   float,
                   double,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(flatten_with_xshape,
                   CPU,
                   ALL_LAYOUT,
                   phi::FlattenWithXShape,
                   float,
                   double,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(flatten,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlattenKernel,
                   float,
                   phi::dtype::float16,
                   double,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(flatten_with_xshape,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlattenWithXShape,
                   float,
                   phi::dtype::float16,
                   double,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}
#endif

#ifdef PADDLE_WITH_XPU
PD_REGISTER_KERNEL(flatten,
                   XPU,
                   ALL_LAYOUT,
                   phi::FlattenKernel,
                   float,
                   phi::dtype::float16,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(flatten_with_xshape,
                   XPU,
                   ALL_LAYOUT,
                   phi::FlattenWithXShape,
                   float,
                   phi::dtype::float16,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}
#endif
