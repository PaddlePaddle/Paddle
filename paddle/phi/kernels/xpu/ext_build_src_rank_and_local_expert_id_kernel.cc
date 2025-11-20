// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void BuildSrcRankAndLocalExpertIdKernel(
    const Context& dev_ctx,
    const DenseTensor& expert_num_global_tensor,
    const std::vector<int64_t>& expert_num_global,
    int64_t num_local_experts,
    DenseTensor* src_rank,
    DenseTensor* local_expert_id) {
  int64_t token_num =
      std::accumulate(expert_num_global.begin(), expert_num_global.end(), 0);

  const int64_t* expert_num_global_tensor_data =
      expert_num_global_tensor.data<int64_t>();

  // Hard coded as ernie-core did.
  int* src_rank_data = dev_ctx.template Alloc<int>(src_rank);
  int* local_expert_id_data = dev_ctx.template Alloc<int>(local_expert_id);

  int r = xpu::build_srcrank_and_local_expert_id(
      dev_ctx.x_context(),
      src_rank_data,
      local_expert_id_data,
      expert_num_global_tensor_data,
      expert_num_global,
      token_num,
      static_cast<int64_t>(expert_num_global.size()),
      num_local_experts);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "build_srcrank_and_local_expert_id");
}

}  // namespace phi

PD_REGISTER_KERNEL(build_src_rank_and_local_expert_id,
                   XPU,
                   ALL_LAYOUT,
                   phi::BuildSrcRankAndLocalExpertIdKernel,
                   int,
                   int64_t) {}
