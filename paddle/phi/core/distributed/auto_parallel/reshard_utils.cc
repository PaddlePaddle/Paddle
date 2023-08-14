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

#include <cstdlib>
#include "glog/logging.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/store/tcp_store.h"

namespace phi {
namespace distributed {
using auto_parallel::str_split;

bool IsDimsMappingShard(const std::vector<int64_t>& dims_mapping) {
  return std::any_of(dims_mapping.begin(),
                     dims_mapping.end(),
                     [](int64_t value) { return value != -1; });
}

bool IsDimsMappingReplicated(const std::vector<int64_t>& dims_mapping) {
  return std::all_of(dims_mapping.begin(),
                     dims_mapping.end(),
                     [](int64_t value) { return value == -1; });
}

std::vector<int64_t> GetCurRankCoordInMesh(const ProcessMesh& process_mesh) {
  const auto& process_shape = process_mesh.shape();
  const auto& process_ids = process_mesh.process_ids();
  int64_t ndims_mesh = process_shape.size();
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

std::map<int64_t, int64_t> GetSplitAxisWithDimsMapping(
    const std::vector<int64_t>& dims_mapping) {
  std::map<int64_t, int64_t> split_axis_to_mesh_axis;
  for (size_t i = 0; i < dims_mapping.size(); ++i) {
    if (dims_mapping[i] != -1) {
      split_axis_to_mesh_axis.emplace(i, dims_mapping[i]);
    }
  }
  return split_axis_to_mesh_axis;
}

int64_t GetCurGlobalRank() {
  const char* cur_rank = std::getenv("PADDLE_TRAINER_ID");
  PADDLE_ENFORCE_NOT_NULL(
      cur_rank,
      phi::errors::NotFound(
          "The environment variable 'PADDLE_TRAINER_ID' cannot be found."));
  return std::atoi(cur_rank);
}

int64_t GetGlobalWorldSize() {
  const char* world_size = std::getenv("PADDLE_TRAINERS_NUM");
  PADDLE_ENFORCE_NOT_NULL(
      world_size,
      phi::errors::NotFound(
          "The environment variable 'PADDLE_TRAINERS_NUM' cannot be found."));
  return std::atoi(world_size);
}

namespace {
std::string GetMasterEndpoint() {
  const char* master_endpoint = std::getenv("PADDLE_MASTER");
  if (!master_endpoint) {
    const char* trainer_endpoints = std::getenv("PADDLE_TRAINER_ENDPOINTS");
    PADDLE_ENFORCE_NOT_NULL(
        trainer_endpoints,
        phi::errors::NotFound("The environment variable "
                              "'PADDLE_TRAINER_ENDPOINTS' cannot be found."));
    return str_split(trainer_endpoints, ",")[0];
  }

  PADDLE_ENFORCE_NOT_NULL(
      master_endpoint,
      phi::errors::NotFound(
          "The environment variable 'PADDLE_MASTER' cannot be found."));
  return master_endpoint;
}

std::string GenUniqueCommKey(const std::vector<int64_t>& process_ids) {
  std::string unique_comm_key = "ReshardGroup";
  for (const auto& id : process_ids) {
    unique_comm_key += "/" + std::to_string(id);
  }
  return unique_comm_key;
}

int64_t GetLocalRankInParticipate(const std::vector<int64_t>& process_ids) {
  int64_t cur_global_rank = GetCurGlobalRank();
  auto iter =
      std::find(process_ids.begin(), process_ids.end(), cur_global_rank);
  return iter - process_ids.begin();
}

}  // namespace

std::string GetMasterAddr() {
  std::string master_endpoint = GetMasterEndpoint();
  return str_split(master_endpoint, ":")[0];
}

uint16_t GetMasterPort() {
  std::string master_endpoint = GetMasterEndpoint();
  return std::stoi(str_split(master_endpoint, ":")[1]);
}

std::shared_ptr<TCPStore> CreateOrGetGlobalTCPStore() {
  std::string host = GetMasterAddr();
  uint16_t port = GetMasterPort();
  int64_t cur_rank = GetCurGlobalRank();
  int64_t world_size = GetGlobalWorldSize();
  bool is_master = (cur_rank == 0);

  static std::shared_ptr<TCPStore> store =
      std::make_shared<TCPStore>(host, port, is_master, world_size);
  return store;
}

CommContext* CreateOrGetCommContext(const DeviceContext& dev_ctx,
                                    const std::vector<int64_t>& process_ids) {
  std::string unique_comm_key = GenUniqueCommKey(process_ids);

  if (!CommContextManager::GetInstance().Has(unique_comm_key)) {
    int64_t world_size = process_ids.size();
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

}  // namespace distributed
}  // namespace phi
