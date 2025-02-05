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

#include "paddle/phi/kernels/fused_softmax_mask_upper_triangle_kernel.h"
#include "paddle/common/errors.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi::fusion {

template <typename T, typename Context>
void FusedSoftmaxMaskFuseUpperTriangleKernel(const Context& dev_ctx,
                                             const DenseTensor& x,
                                             DenseTensor* out) {
  bool is_gpu_place = dev_ctx.GetPlace().GetType() == phi::AllocationType::GPU;
  PADDLE_ENFORCE_EQ(is_gpu_place,
                    true,
                    common::errors::Unimplemented(
                        "Softmax mask fuse op only supports GPU now."));
}

}  // namespace phi::fusion

PD_REGISTER_KERNEL(fused_softmax_mask_upper_triangle,
                   CPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedSoftmaxMaskFuseUpperTriangleKernel,
                   float,
                   double) {}
