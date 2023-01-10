// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(PADDLE_WITH_GLOO)
#include "gloo/rendezvous/prefix_store.h"

#include "paddle/phi/core/distributed/gloo_comm_context.h"
#include "paddle/phi/core/distributed/gloo_utils.h"
#include "paddle/phi/core/distributed/store/gloo_store.h"
#endif

#include "paddle/phi/core/distributed/comm_context_manager.h"

#include <memory>
#include <string>

#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/distributed/store/store.h"
#include "paddle/phi/core/enforce.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#endif

namespace phi {
namespace distributed {

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
void CommContextManager::CreateNCCLCommContext(
    const std::shared_ptr<Store>& store,
    int dev_id,
    int ring_id,
    int rank,
    int size) {
  phi::backends::gpu::SetDeviceId(dev_id);
  ncclUniqueId nccl_id;
  if (rank == 0) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclGetUniqueId(&nccl_id));
  }

  std::string unique_key = "NCCLCommContext/" + std::to_string(ring_id);
  if (rank == 0) {
    std::vector<uint8_t> nccl_id_wrapper(
        reinterpret_cast<uint8_t*>(&nccl_id),
        reinterpret_cast<uint8_t*>(&nccl_id) + NCCL_UNIQUE_ID_BYTES);
    store->set(unique_key, nccl_id_wrapper);
  } else {
    const auto& nccl_id_wrapper = store->get(unique_key);
    std::memcpy(&nccl_id, nccl_id_wrapper.data(), nccl_id_wrapper.size());
  }

  auto nccl_comm_context =
      std::make_unique<NCCLCommContext>(rank, size, nccl_id);
  auto& comm_context_manager = CommContextManager::GetInstance();
  comm_context_manager.SetStore(store);
  comm_context_manager.Emplace(ring_id, std::move(nccl_comm_context));
}
#endif

#if defined(PADDLE_WITH_GLOO)
void CommContextManager::CreateGlooCommContext(
    const std::shared_ptr<Store>& store, int ring_id, int rank, int size) {
  GlooStore store_wrapper(store);
  auto gloo_store = std::make_shared<gloo::rendezvous::PrefixStore>(
      std::to_string(ring_id), store_wrapper);

  auto gloo_device = CreateGlooDevice();

  auto gloo_comm_context =
      std::make_unique<GlooCommContext>(rank, size, gloo_store, gloo_device);
  auto& comm_context_manager = CommContextManager::GetInstance();
  // set actual store to manager
  comm_context_manager.SetStore(store);
  comm_context_manager.Emplace(ring_id, std::move(gloo_comm_context));
}
#endif

CommContext* CommContextManager::Emplace(
    int ring_id, std::unique_ptr<CommContext> comm_context) {
  PADDLE_ENFORCE_EQ(
      id_to_comm_context_.find(ring_id),
      id_to_comm_context_.end(),
      errors::AlreadyExists("Ring id %d already exists in the map.", ring_id));
  id_to_comm_context_.emplace(ring_id, std::move(comm_context));
  return id_to_comm_context_.at(ring_id).get();
}

CommContext* CommContextManager::Get(int ring_id) const {
  PADDLE_ENFORCE_NE(
      id_to_comm_context_.find(ring_id),
      id_to_comm_context_.end(),
      errors::NotFound("Can not find ring id %d in map.", ring_id));

  return id_to_comm_context_.at(ring_id).get();
}

bool CommContextManager::Has(int ring_id) const {
  return id_to_comm_context_.find(ring_id) != id_to_comm_context_.end();
}

}  // namespace distributed
}  // namespace phi
