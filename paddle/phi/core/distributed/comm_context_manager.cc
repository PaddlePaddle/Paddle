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

#include "paddle/phi/core/distributed/comm_context_manager.h"

#include <memory>
#include <string>
#include "glog/logging.h"

#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/distributed/store/store.h"
#include "paddle/phi/core/enforce.h"

#if defined(PADDLE_WITH_GLOO)
#include <gloo/rendezvous/prefix_store.h>
#include "paddle/phi/core/distributed/gloo_comm_context.h"
#include "paddle/phi/core/distributed/gloo_utils.h"
#include "paddle/phi/core/distributed/store/gloo_store.h"
#endif

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/phi/core/distributed/xccl_comm_context.h"
#endif

namespace phi {
namespace distributed {

int CommContextManager::device_id = -1;

void CommContextManager::SetDeviceId(int dev_id) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  phi::backends::gpu::SetDeviceId(dev_id);
  CommContextManager::device_id = dev_id;
#endif
}

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
void CommContextManager::CreateNCCLCommContext(
    const std::shared_ptr<Store>& store,
    const std::string& unique_comm_key,
    int rank,
    int size,
    const std::string& hash_key) {
  auto& comm_context_manager = CommContextManager::GetInstance();
  if (comm_context_manager.Has(unique_comm_key)) {
    return;
  }
  ncclUniqueId nccl_id;
  if (rank == 0) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclGetUniqueId(&nccl_id));
  }

  std::string unique_key = "NCCLCommContext/" + unique_comm_key + hash_key;
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
  if (CommContextManager::device_id != -1) {
    std::unique_ptr<phi::GPUContext> dev_ctx(
        new phi::GPUContext(phi::GPUPlace(CommContextManager::device_id)));
    dev_ctx->SetAllocator(phi::memory_utils::GetAllocator(
        CommContextManager::device_id, dev_ctx->stream()));
    dev_ctx->SetHostAllocator(phi::memory_utils::GetHostAllocator());
    dev_ctx->SetZeroAllocator(
        phi::memory_utils::GetZeroAllocator(CommContextManager::device_id));
    dev_ctx->SetHostZeroAllocator(phi::memory_utils::GetHostZeroAllocator());
    dev_ctx->SetPinnedAllocator(phi::memory_utils::GetPinnedAllocator());
    dev_ctx->PartialInitWithAllocator();
    auto compute_event =
        phi::memory_utils::GetCudaEvent(CommContextManager::device_id);
    auto comm_event =
        phi::memory_utils::GetCudaEvent(CommContextManager::device_id);

    nccl_comm_context->SetDevContext(std::move(dev_ctx));
    nccl_comm_context->SetComputeEvent(std::move(compute_event));
    nccl_comm_context->SetCommEvent(std::move(comm_event));
  }

  comm_context_manager.SetStore(store);
  comm_context_manager.Emplace(unique_comm_key, std::move(nccl_comm_context));
}
#endif

#if defined(PADDLE_WITH_GLOO)
void CommContextManager::CreateGlooCommContext(
    const std::shared_ptr<Store>& store,
    const std::string& unique_comm_key,
    int rank,
    int size) {
  GlooStore store_wrapper(store);
  auto gloo_store = std::make_shared<gloo::rendezvous::PrefixStore>(
      unique_comm_key, store_wrapper);

  auto gloo_device = CreateGlooDevice();

  auto gloo_comm_context =
      std::make_unique<GlooCommContext>(rank, size, gloo_store, gloo_device);
  auto& comm_context_manager = CommContextManager::GetInstance();
  // set actual store to manager
  comm_context_manager.SetStore(store);
  comm_context_manager.Emplace(unique_comm_key, std::move(gloo_comm_context));
}
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
void CommContextManager::CreateXCCLCommContext(
    const std::shared_ptr<Store>& store,
    const std::string& unique_comm_key,
    const std::string& device_type,
    int rank,
    int size) {
  phi::ccl::CCLRootId xccl_root_id;
  if (rank == 0) {
    phi::DeviceManager::CCLGetUniqueId(device_type, &xccl_root_id);
  }

  std::string unique_key = "XCCLCommContext/" + unique_comm_key;
  if (rank == 0) {
    store->set(unique_key, xccl_root_id);
  } else {
    xccl_root_id = store->get(unique_key);
  }
  VLOG(3) << "init xccl rank: " << rank << ", nranks: " << size
          << ", unique_comm_key: " << unique_comm_key << ", xccl uniqueid: "
          << phi::ccl::SerializeXCCLUniqueId(xccl_root_id);
  auto xccl_comm_context =
      std::make_unique<XCCLCommContext>(device_type, rank, size, xccl_root_id);
  auto& comm_context_manager = CommContextManager::GetInstance();
  comm_context_manager.SetStore(store);
  comm_context_manager.Emplace(unique_comm_key, std::move(xccl_comm_context));
}
#endif

CommContext* CommContextManager::Emplace(
    const std::string& unique_comm_key,
    std::unique_ptr<CommContext> comm_context) {
  PADDLE_ENFORCE_EQ(
      id_to_comm_context_.find(unique_comm_key),
      id_to_comm_context_.end(),
      errors::AlreadyExists("The unique key %s already exists in the map.",
                            unique_comm_key));
  id_to_comm_context_.emplace(unique_comm_key, std::move(comm_context));
  return id_to_comm_context_.at(unique_comm_key).get();
}

CommContext* CommContextManager::Get(const std::string& unique_comm_key) const {
  PADDLE_ENFORCE_NE(
      id_to_comm_context_.find(unique_comm_key),
      id_to_comm_context_.end(),
      errors::NotFound("Can not find unique key %s in map.", unique_comm_key));

  return id_to_comm_context_.at(unique_comm_key).get();
}

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
int CommContextManager::GetRingId(const ncclComm_t& comm) const {
  for (auto iter = id_to_comm_context_.begin();
       iter != id_to_comm_context_.end();
       ++iter) {
    if (static_cast<phi::distributed::NCCLCommContext*>(iter->second.get())
            ->GetNcclComm() == comm) {
      return std::stoi(iter->first);
    }
  }
  return -1;
}
#endif

bool CommContextManager::Has(const std::string& unique_comm_key) const {
  return id_to_comm_context_.find(unique_comm_key) != id_to_comm_context_.end();
}

}  // namespace distributed
}  // namespace phi
