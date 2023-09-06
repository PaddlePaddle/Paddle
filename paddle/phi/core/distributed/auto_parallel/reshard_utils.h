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

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/visit_type.h"

namespace phi {
class DeviceContext;

namespace distributed {
class ProcessMesh;

bool IsDimsMappingShard(const std::vector<int64_t>& dims_mapping);

bool IsDimsMappingReplicated(const std::vector<int64_t>& dims_mapping);

// Get the coordinate of cur rank in process mesh. For example, the process mesh
// is [[0, 1], [2, 3], [4, 5], [6, 7]], if the current rank is 4, then will
// return [2, 0]; if the current rank is 3, then will return [1, 1].
std::vector<int64_t> GetCurRankCoordInMesh(const ProcessMesh& process_mesh);

// If the index i's value in dims_mapping is x ( x != -1), means the ith axis of
// tensor need be split by xth axis of process_mesh. The function analyze the
// input vector, return a key-value map of tensor_split_axis and
// process_mesh_split_axis.
// For example, if dims_mapping is [-1, 1, -1, 0], will return {1: 1, 3: 0}.
std::map<int64_t, int64_t> GetSplitAxisWithDimsMapping(
    const std::vector<int64_t>& dims_mapping);

// If given a number, balance split it to multiple pieces.
// For example, the input value is 12, split it to 5 pieces, then return
// {3, 3, 2, 2, 2}.
std::vector<int64_t> BalancedSplit(int64_t total_nums, int64_t num_of_pieces);

// Create a comm context of the input process_ids. Once the newly comm context
// created, it will be cached in the global instance, and get from the global
// cache later. If the input dev_ctx is GPU, then nccl comm context will be
// created. If the input dev_ctx is CPU, then gloo comm context will be created.
CommContext* CreateOrGetCommContext(const DeviceContext& dev_ctx,
                                    const std::vector<int64_t>& process_ids);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#define RESHARD_FUNCTOR_IMPL(dev_ctx, fn_name, dtype, ...)            \
  do {                                                                \
    if (phi::CPUContext::classof(dev_ctx)) {                          \
      PD_VISIT_FLOATING_AND_INTEGRAL_TYPES(                           \
          dtype, #fn_name, ([&] {                                     \
            fn_name<data_t>(static_cast<const CPUContext&>(*dev_ctx), \
                            __VA_ARGS__);                             \
          }));                                                        \
    } else if (phi::GPUContext::classof(dev_ctx)) {                   \
      PD_VISIT_FLOATING_AND_INTEGRAL_TYPES(                           \
          dtype, #fn_name, ([&] {                                     \
            fn_name<data_t>(static_cast<const GPUContext&>(*dev_ctx), \
                            __VA_ARGS__);                             \
          }));                                                        \
    } else {                                                          \
      PADDLE_THROW(phi::errors::Unimplemented(                        \
          "The %s in reshard only supported on CPU and GPU for now.", \
          #fn_name));                                                 \
    }                                                                 \
  } while (0)
#else
#define RESHARD_FUNCTOR_IMPL(dev_ctx, fn_name, dtype, ...)                \
  do {                                                                    \
    if (phi::CPUContext::classof(dev_ctx)) {                              \
      PD_VISIT_FLOATING_AND_INTEGRAL_TYPES(                               \
          dtype, #fn_name, ([&] {                                         \
            fn_name<data_t>(static_cast<const CPUContext&>(*dev_ctx),     \
                            __VA_ARGS__);                                 \
          }));                                                            \
    } else {                                                              \
      PADDLE_THROW(phi::errors::Unimplemented(                            \
          "The %s in reshard only supported on CPU for now.", #fn_name)); \
    }                                                                     \
  } while (0)
#endif

#define RESHARD_FUNCTOR_WITH_COMM(dev_ctx, fn_name, dtype, process_ids, ...) \
  do {                                                                       \
    auto* comm_context = CreateOrGetCommContext(*dev_ctx, process_ids);      \
    dev_ctx->SetCommContext(comm_context);                                   \
    RESHARD_FUNCTOR_IMPL(dev_ctx, fn_name, dtype, __VA_ARGS__);              \
  } while (0)

#define RESHARD_FUNCTOR(dev_ctx, fn_name, dtype, ...)           \
  do {                                                          \
    RESHARD_FUNCTOR_IMPL(dev_ctx, fn_name, dtype, __VA_ARGS__); \
  } while (0)

}  // namespace distributed
}  // namespace phi
