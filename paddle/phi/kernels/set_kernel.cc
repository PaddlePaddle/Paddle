// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/kernels/set_kernel.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void SetKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const DenseTensor& source,
               const std::vector<int64_t>& dims,
               const std::vector<int64_t>& stride,
               int64_t offset,
               DenseTensor* out) {
  auto meta = out->meta();
  meta.dims = DDim(dims.data(), static_cast<int>(dims.size()));
  meta.strides = DDim(stride.data(), static_cast<int>(stride.size()));
  meta.offset = offset;
  if (x.IsSharedWith(source)) {
    out->set_meta(meta);
  } else {
    // reset holder to nullptr
    out->clear();
    *out = DenseTensor{source.Holder(), meta};
  }
  out->ShareInplaceVersionCounterWith(x);
}

}  // namespace phi

PD_REGISTER_KERNEL(set,
                   CPU,
                   ALL_LAYOUT,
                   phi::SetKernel,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(set,
                   GPU,
                   ALL_LAYOUT,
                   phi::SetKernel,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#endif
