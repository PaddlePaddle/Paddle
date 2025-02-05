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

#include "paddle/phi/kernels/limit_by_capacity_kernel.h"
#include "paddle/common/errors.h"
#include "paddle/phi/core/kernel_registry.h"

#if defined(PADDLE_WITH_GLOO)
#include "paddle/phi/core/distributed/gloo_comm_context.h"
#endif
namespace phi {

template <typename T, typename Context>
void LimitByCapacityKernel(const Context& dev_ctx,
                           const DenseTensor& expert_count,
                           const DenseTensor& capacity,
                           int n_worker,
                           DenseTensor* Out) {
  PADDLE_THROW(common::errors::Unimplemented(
      "limit_by_capacity is not supported on CPU."));
}

}  // namespace phi

PD_REGISTER_KERNEL(limit_by_capacity,
                   CPU,
                   ALL_LAYOUT,
                   phi::LimitByCapacityKernel,
                   int,
                   int64_t) {}
