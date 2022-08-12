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

#include "paddle/phi/kernels/memcpy_d2h_kernel.h"

#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/kernel_registry.h"
// #include "paddle/phi/common/place.h"

namespace phi {

template <typename T, typename Context>
void MemcpyD2HKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     int dst_place_type,
                     DenseTensor* out) {
  PADDLE_ENFORCE_GE(
      dst_place_type,
      0,
      errors::OutOfRange("dst_place_type only support 0-1, but got: %d",
                         dst_place_type));
  PADDLE_ENFORCE_LE(
      dst_place_type,
      1,
      errors::OutOfRange("dst_place_type only support 0-1, but got: %d",
                         dst_place_type));

  // Copy will set the stream of the tensor while setting blocking to false
  switch (dst_place_type) {
    case 0:
      Copy(dev_ctx, x, CPUPlace(), false, out);
      break;
    case 1:
      Copy(dev_ctx, x, GPUPinnedPlace(), false, out);
      break;

    default:
      break;
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(memcpy_d2h,
                   CPU,
                   ALL_LAYOUT,
                   phi::MemcpyD2HKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>,
                   phi::dtype::float16,
                   int16_t) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(memcpy_d2h,
                   GPU,
                   ALL_LAYOUT,
                   phi::MemcpyD2HKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>,
                   phi::dtype::float16,
                   int16_t) {}
#endif

#ifdef PADDLE_WITH_XPU
PD_REGISTER_KERNEL(memcpy_d2h,
                   XPU,
                   ALL_LAYOUT,
                   phi::MemcpyD2HKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>,
                   phi::dtype::float16,
                   int16_t) {}
#endif
