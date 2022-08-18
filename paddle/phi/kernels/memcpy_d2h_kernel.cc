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

#include <vector>

#include "glog/logging.h"

#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

static constexpr size_t WAIT_THRESHOLD = 64 * 1024;

template <typename T, typename Context>
void MemcpyD2HKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     int dst_place_type,
                     DenseTensor* out) {
  // Copy will set the stream of the tensor while setting blocking to false
  // it also handles asynchronous small data(<64kb) copy bewteen CpuPlace and
  // CudaPlace so no need to do anything special
  switch (dst_place_type) {
    case 0:
      Copy(dev_ctx, x, CPUPlace(), false, out);
      break;
    case 1:
      Copy(dev_ctx, x, GPUPinnedPlace(), false, out);
      break;

    default:
      PADDLE_THROW(errors::OutOfRange(
          "dst_place_type only support 0-1, but got: %d", dst_place_type));
      break;
  }
}

template <typename T, typename Context>
void MemcpyD2HMultiIOKernel(const Context& dev_ctx,
                            const std::vector<const DenseTensor*>& array,
                            int dst_place_type,
                            std::vector<DenseTensor*> out_array) {
  VLOG(10) << "using MemcpyD2HMultiIOKernel";
  out_array.clear();
  out_array.resize(array.size());

  for (size_t i = 0; i < array.size(); i++) {
    const auto& x = *(out_array[i]);
    VLOG(10) << i << " " << out_array[i] << " " << x.initialized();
    MemcpyD2HKernel<T, Context>(dev_ctx, x, dst_place_type, out_array[i]);
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

PD_REGISTER_KERNEL(memcpy_d2h_multi_io,
                   CPU,
                   ALL_LAYOUT,
                   phi::MemcpyD2HMultiIOKernel,
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

PD_REGISTER_KERNEL(memcpy_d2h_multi_io,
                   GPU,
                   ALL_LAYOUT,
                   phi::MemcpyD2HMultiIOKernel,
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

PD_REGISTER_KERNEL(memcpy_d2h_multi_io,
                   XPU,
                   ALL_LAYOUT,
                   phi::MemcpyD2HMultiIOKernel,
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
