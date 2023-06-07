// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename Context>
void RealStridedKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       DenseTensor* out) {
  DDim stride = out->stride();
  for (int i = 0; i < stride.size(); i++) {
    stride[i] = out->stride()[i] * 2;
  }
  out->set_stride(stride);
  out->ResetHolder(x.Holder());
}

template <typename Context>
void ImagStridedKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       DenseTensor* out) {
  DDim stride = out->stride();
  for (int i = 0; i < stride.size(); i++) {
    stride[i] = out->stride()[i] * 2;
  }
  out->set_stride(stride);
  out->set_offset(phi::SizeOf(out->dtype()));
  out->ResetHolder(x.Holder());
}

}  // namespace phi
PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE_EXCEPT_CUSTOM(real,
                                                       STRIDED,
                                                       phi::RealStridedKernel) {
}

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE_EXCEPT_CUSTOM(imag,
                                                       STRIDED,
                                                       phi::ImagStridedKernel) {
}
