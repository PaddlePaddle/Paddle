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

#include "paddle/phi/kernels/strided_slice_kernel.h"

#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void StridedSliceKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const std::vector<int>& axes,
                        const IntArray& starts,
                        const IntArray& ends,
                        const IntArray& strides,
                        DenseTensor* out) {
  DenseTensor& xx = const_cast<DenseTensor&>(x);
  out->can_not_uses = xx.can_not_uses;
  if (*out->canNotUse == false) {
    *out->canNotUse = *xx.canNotUse;
  }
  xx.can_not_uses->insert(xx.canNotUse);

  xx.can_not_uses->insert(out->canNotUse);
  std::vector<int> infer_flags(axes.size(), 1);
  std::vector<int> decrease_axis;
  StridedSliceRawKernel<T, Context>(
      dev_ctx, x, axes, starts, ends, strides, infer_flags, decrease_axis, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(strided_slice,
                   CPU,
                   ALL_LAYOUT,
                   phi::StridedSliceKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(strided_slice,
                   GPU,
                   ALL_LAYOUT,
                   phi::StridedSliceKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#endif
#if defined(PADDLE_WITH_XPU)
PD_REGISTER_KERNEL(strided_slice,
                   XPU,
                   ALL_LAYOUT,
                   phi::StridedSliceKernel,
                   int,
                   int16_t,
                   float,
                   phi::dtype::float16) {}
#endif
