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

#include "paddle/phi/kernels/impl/bilateral_slice_kernel_impl.h"

#include <memory>
#include <string>
#include <vector>

#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void BilateralSliceKernel(const Context &dev_ctx,
                          const DenseTensor &x UNUSED,
                          const DenseTensor &grid UNUSED,
                          const DenseTensor &guide UNUSED,
                          bool has_offset UNUSED,
                          DenseTensor *out UNUSED) {
  PADDLE_ENFORCE_EQ(
      dev_ctx.GetPlace().GetType() == phi::AllocationType::GPU,
      true,
      phi::errors::Unimplemented("BilateralSlice only supports GPU now."));
}
}  // namespace phi

PD_REGISTER_KERNEL(bilateral_slice,
                   CPU,
                   ALL_LAYOUT,
                   phi::BilateralSliceKernel,
                   float,
                   double) {}
