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
// avoid winsock conflict
#ifdef _WIN32
#include <gloo/common/win.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <gloo/rendezvous/context.h>
#include <gloo/rendezvous/prefix_store.h>
#include <gloo/rendezvous/store.h>
#include <gloo/transport/tcp/device.h>

#include "paddle/phi/core/distributed/gloo_comm_context.h"
#include "paddle/phi/core/distributed/store/tcp_utils.h"
#include "paddle/phi/core/errors.h"
#endif

#include "paddle/phi/core/distributed/comm_context_manager.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
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
class GlooStore : public gloo::rendezvous::Store {
 public:
  explicit GlooStore(const std::shared_ptr<phi::distributed::Store>& store)
      : store_(store) {}

  ~GlooStore() = default;

  std::vector<char> get(const std::string& key) override {
    VLOG(3) << "GlooStore::get";
    auto value = store_->get(key);
    return std::vector<char>(value.begin(), value.end());
  }

  void wait(const std::vector<std::string>& keys) override {
    VLOG(3) << "GlooStore::wait";
    for (auto& key : keys) {
      store_->wait(key);
    }
  }

  void set(const std::string& key, const std::vector<char>& value) override {
    VLOG(3) << "GlooStore::set";
    std::vector<uint8_t> tmp(value.begin(), value.end());
    store_->set(key, tmp);
  }

  void wait(const std::vector<std::string>& keys,
            const std::chrono::milliseconds& timeout) override {
    VLOG(3) << "GlooStore::wait";
    for (auto& key : keys) {
      store_->wait(key);
    }
  }

 protected:
  std::shared_ptr<phi::distributed::Store> store_;
};

std::shared_ptr<gloo::transport::Device> CreateDeviceForInterface(
    const std::string& ifname) {
  gloo::transport::tcp::attr attr;
  attr.iface = ifname;
  return gloo::transport::tcp::CreateDevice(attr);
}

std::shared_ptr<gloo::transport::Device> CreateDeviceForHostname(
    const std::string& hostname) {
  gloo::transport::tcp::attr attr;
  attr.hostname = hostname;
  return gloo::transport::tcp::CreateDevice(attr);
}

std::shared_ptr<gloo::transport::Device> CreateDefaultDevice() {
  std::array<char, HOST_NAME_MAX> hostname;
  auto ret = ::gethostname(hostname.data(), HOST_NAME_MAX);
  PADDLE_ENFORCE_EQ(
      ret,
      0,
      phi::errors::Fatal("Get hostname error for createDefaultDevice."));
  ::addrinfo* result;
  result = phi::distributed::tcputils::get_addr_info(
      hostname.data(), "", 0, AF_UNSPEC);
  ::addrinfo* cur;
  for (cur = result; cur != nullptr; cur = cur->ai_next) {
    phi::distributed::SocketType socket =
        ::socket(cur->ai_family, cur->ai_socktype, cur->ai_protocol);
    if (socket == -1) {
      continue;
    }
    ret = ::bind(socket, cur->ai_addr, cur->ai_addrlen);
#ifdef _WIN32
    closesocket(socket);
#else
    close(socket);
#endif
    if (ret == -1) {
      continue;
    }
    break;
  }
  freeaddrinfo(result);
  if (cur != nullptr) {
    return CreateDeviceForHostname(hostname.data());
  }
  return CreateDeviceForHostname("127.0.0.1");
}

void CommContextManager::CreateGlooCommContext(
    const std::shared_ptr<Store>& store, int ring_id, int rank, int size) {
  char* ifname = std::getenv("GLOO_SOCKET_IFNAME");
  auto gloo_device = ifname && std::strlen(ifname) > 1
                         ? CreateDeviceForInterface(std::string(ifname))
                         : CreateDefaultDevice();

  GlooStore store_wrapper(store);
  auto gloo_store =
      gloo::rendezvous::PrefixStore(std::to_string(ring_id), store_wrapper);

  auto gloo_context = std::make_shared<gloo::rendezvous::Context>(rank, size);
  gloo_context->connectFullMesh(gloo_store, gloo_device);

  auto gloo_comm_context =
      std::make_unique<GlooCommContext>(rank, size, gloo_context);
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
