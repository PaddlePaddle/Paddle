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

#include "paddle/phi/kernels/broadcast_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

#if defined(PADDLE_WITH_GLOO)
#include "paddle/phi/core/distributed/gloo_comm_context.h"
#endif

namespace phi {

template <typename T, typename Context>
void BroadcastKernel(const Context& dev_ctx,
        const DenseTensor& x,
        int ring_id,
        int root,
        bool use_calc_stream,
        DenseTensor* out) {
#if defined(PADDLE_WITH_GLOO)
    LOG(INFO) << "broadcast kernel gloo begin";
    dev_ctx.template Alloc<T>(out);
    auto comm_context = static_cast<distributed::GlooCommContext*>(dev_ctx.GetCommContext());
    comm_context->Broadcast(out, x, root);
#else
    PADDLE_THROW(errors::Unavailable(
        "PaddlePaddle should compile with GLOO by setting WITH_GLOO=ON"));
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(broadcast,
                   CPU,
                   ALL_LAYOUT,
                   phi::BroadcastKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16) {}
