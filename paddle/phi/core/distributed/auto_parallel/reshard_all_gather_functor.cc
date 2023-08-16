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

#include "paddle/phi/core/distributed/auto_parallel/reshard_all_gather_functor.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard_utils.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/kernels/all_gather_kernel.h"

namespace phi {
namespace distributed {

DenseTensor ReshardAllGatherFunctor(DeviceContext* dev_ctx,
                                    const DenseTensor& input,
                                    const std::vector<int64_t>& process_ids) {
  DenseTensor out;

  int64_t world_size = process_ids.size();
  auto* comm_context = CreateOrGetCommContext(*dev_ctx, process_ids);
  dev_ctx->SetCommContext(comm_context);

  if (phi::CPUContext::classof(dev_ctx)) {
    PD_VISIT_FLOATING_AND_INTEGRAL_TYPES(
        input.dtype(), "AllGather", ([&] {
          AllGather<data_t>(static_cast<const CPUContext&>(*dev_ctx),
                            input,
                            world_size,
                            &out);
        }));
    return out;
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (phi::GPUContext::classof(dev_ctx)) {
    PD_VISIT_FLOATING_AND_INTEGRAL_TYPES(
        input.dtype(), "AllGather", ([&] {
          AllGather<data_t>(static_cast<const GPUContext&>(*dev_ctx),
                            input,
                            world_size,
                            &out);
        }));
    return out;
  }
#endif
  PADDLE_THROW(phi::errors::Unimplemented(
      "The all_gather in reshard only supported on CPU and GPU for now."));
}

}  // namespace distributed
}  // namespace phi
