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

#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/kernel_registry.h"

#if defined(PADDLE_WITH_GLOO)
#include <gloo/barrier.h>
#include "paddle/phi/core/distributed/gloo_comm_context.h"
#endif

namespace phi {

template <typename T, typename Context>
void BarrierKernel(const Context& dev_ctx,
                   const DenseTensor& x_in,
                   int ring_id,
                   DenseTensor* out) {
#if defined(PADDLE_WITH_GLOO)
  const auto& comm_context_manager =
      phi::distributed::CommContextManager::GetInstance();
  if (comm_context_manager.Has(std::to_string(ring_id))) {
    auto* comm_context = static_cast<phi::distributed::GlooCommContext*>(
        comm_context_manager.Get(std::to_string(ring_id)));
    comm_context->Barrier();
  } else {
    PADDLE_THROW(phi::errors::Unavailable(
        "You must initialize the gloo environment first to use it."));
  }
#else
  PADDLE_THROW(phi::errors::Unavailable(
      "PaddlePaddle should compile with GLOO by setting WITH_GLOO=ON"));
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(barrier, CPU, ALL_LAYOUT, phi::BarrierKernel, int) {}
