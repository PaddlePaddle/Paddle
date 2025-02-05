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

#include "paddle/phi/kernels/assign_pos_kernel.h"
#include "paddle/common/errors.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void AssignPosKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& cum_count,
                     const DenseTensor& eff_num_len,
                     DenseTensor* out) {
  PADDLE_THROW(common::errors::Unavailable(
      "Do not support assign pos op for cpu kernel now."));
}

}  // namespace phi

PD_REGISTER_KERNEL(
    assign_pos, CPU, ALL_LAYOUT, phi::AssignPosKernel, int, int64_t) {}
