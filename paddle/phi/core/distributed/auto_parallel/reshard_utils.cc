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

#include "paddle/phi/core/distributed/auto_parallel/reshard_utils.h"

#include "glog/logging.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/store/store_utils.h"

namespace phi {
namespace distributed {

namespace {
std::string GenUniqueCommKey(const std::vector<int64_t>& process_ids) {
  std::string unique_comm_key = "ReshardGroup";
  for (const auto& id : process_ids) {
    unique_comm_key += "/" + std::to_string(id);
  }
  return unique_comm_key;
}
}  // namespace

int64_t GetLocalRankInParticipate(const std::vector<int64_t>& process_ids,
                                  int64_t global_rank) {
  if (global_rank == -1) {
    global_rank = GetCurGlobalRank();
  }
  auto iter = std::find(process_ids.begin(), process_ids.end(), global_rank);
  PADDLE_ENFORCE_NE(
      iter,
      process_ids.end(),
      phi::errors::NotFound("Global rank %lld cannot be found in process_mesh",
                            global_rank));
  return iter - process_ids.begin();
}

std::vector<int64_t> GetCurRankCoordInMesh(const ProcessMesh& process_mesh) {
  const auto& process_shape = process_mesh.shape();
  const auto& process_ids = process_mesh.process_ids();
  int64_t ndims_mesh = static_cast<int64_t>(process_shape.size());
  int64_t cur_global_rank = GetCurGlobalRank();

  VLOG(3) << "Searching current global rank " << cur_global_rank
          << " in process_mesh " << process_mesh;

  auto iter =
      std::find(process_ids.begin(), process_ids.end(), cur_global_rank);
  PADDLE_ENFORCE_NE(
      iter,
      process_ids.end(),
      phi::errors::NotFound("Rank %lld cannot be found in process_mesh",
                            cur_global_rank));

  int64_t flat_idx_in_mesh = iter - process_ids.begin();

  std::vector<int64_t> coord(ndims_mesh, -1);
  for (int64_t i = ndims_mesh - 1; i >= 0; --i) {
    coord[i] = flat_idx_in_mesh % process_shape[i];
    flat_idx_in_mesh /= process_shape[i];
  }
  return coord;
}

CommContext* CreateOrGetCommContext(const DeviceContext& dev_ctx,
                                    const std::vector<int64_t>& process_ids) {
  std::string unique_comm_key = GenUniqueCommKey(process_ids);

  if (!CommContextManager::GetInstance().Has(unique_comm_key)) {
    int64_t world_size = static_cast<int64_t>(process_ids.size());
    int64_t rank = GetLocalRankInParticipate(process_ids);
    VLOG(3) << "local world size: " << world_size << " local rank: " << rank;

    auto store = CreateOrGetGlobalTCPStore();
    if (phi::CPUContext::classof(&dev_ctx)) {
#if defined(PADDLE_WITH_GLOO)
      CommContextManager::CreateGlooCommContext(
          store, unique_comm_key, rank, world_size);
#else
      PADDLE_THROW(phi::errors::Unimplemented(
          "Cannot use gloo on CPU, please turn PADDLE_WITH_GLOO flag on."));
#endif
    } else if (phi::CustomContext::classof(&dev_ctx)) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
      CommContextManager::CreateXCCLCommContext(
          store,
          unique_comm_key,
          dev_ctx.GetPlace().GetDeviceType(),
          rank,
          world_size);
#endif
    } else {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
      if (phi::GPUContext::classof(&dev_ctx)) {
        CommContextManager::CreateNCCLCommContext(
            store, unique_comm_key, rank, world_size);
      }
#else
      PADDLE_THROW(phi::errors::Unimplemented(
          "CommContext is only supported on CPU and GPU for now, other devices "
          "will be supported later."));
#endif
    }
  }

  auto* comm_context = CommContextManager::GetInstance().Get(unique_comm_key);
  return comm_context;
}

std::map<int, int64_t> GetSplitAxisWithDimsMapping(
    const std::vector<int64_t>& dims_mapping) {
  std::map<int, int64_t> split_axis_to_mesh_axis;
  for (size_t i = 0; i < dims_mapping.size(); ++i) {
    if (dims_mapping[i] != -1) {
      split_axis_to_mesh_axis.emplace(i, dims_mapping[i]);
    }
  }
  return split_axis_to_mesh_axis;
}

std::vector<int64_t> BalancedSplit(int64_t total_nums, int64_t num_of_pieces) {
  std::vector<int64_t> result(num_of_pieces, total_nums / num_of_pieces);
  int64_t remain_nums = total_nums % num_of_pieces;
  for (int64_t i = 0; i < remain_nums; ++i) {
    result[i] += 1;
  }
  return result;
}

bool IsCurRankInMesh(const ProcessMesh& process_mesh) {
  int64_t cur_global_rank = GetCurGlobalRank();
  const auto& process_ids = process_mesh.process_ids();
  return (std::find(process_ids.begin(), process_ids.end(), cur_global_rank) !=
          process_ids.end());
}

}  // namespace distributed
}  // namespace phi
