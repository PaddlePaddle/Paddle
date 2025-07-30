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

#include "paddle/phi/core/distributed/store/store_utils.h"

#include <cstdlib>

// the <winsock2.h> needs to be included before <winsock.h>, otherwise
// there will be symbol redefinition error on windows
#include "paddle/phi/core/distributed/store/tcp_store.h"

#include "paddle/phi/core/distributed/auto_parallel/utils.h"

namespace phi::distributed {
using auto_parallel::str_split;

namespace {
std::string GetMasterEndpoint() {
  const char* master_endpoint = std::getenv("PADDLE_MASTER");
  if (!master_endpoint) {
    const char* trainer_endpoints = std::getenv("PADDLE_TRAINER_ENDPOINTS");
    PADDLE_ENFORCE_NOT_NULL(trainer_endpoints,
                            common::errors::NotFound(
                                "The environment variable "
                                "'PADDLE_TRAINER_ENDPOINTS' cannot be found."));
    return str_split(trainer_endpoints, ",")[0];
  }

  PADDLE_ENFORCE_NOT_NULL(
      master_endpoint,
      common::errors::NotFound(
          "The environment variable 'PADDLE_MASTER' cannot be found."));
  return master_endpoint;
}
}  // namespace

int64_t GetCurGlobalRank() {
  const char* cur_rank = std::getenv("PADDLE_TRAINER_ID");
  if (cur_rank == nullptr) {
    return 0;
  }
  return std::atoi(cur_rank);
}

int64_t GetGlobalWorldSize() {
  const char* world_size = std::getenv("PADDLE_TRAINERS_NUM");
  if (world_size == nullptr) {
    return 1;
  }
  return std::atoi(world_size);
}

std::string GetMasterAddr() {
  std::string master_endpoint = GetMasterEndpoint();
  return str_split(master_endpoint, ":")[0];
}

uint16_t GetMasterPort() {
  std::string master_endpoint = GetMasterEndpoint();
  return std::stoi(str_split(master_endpoint, ":")[1]);
}

std::shared_ptr<Store> CreateOrGetGlobalTCPStore() {
  std::string host = GetMasterAddr();
  uint16_t port = GetMasterPort();
  int64_t cur_rank = GetCurGlobalRank();
  int64_t world_size = GetGlobalWorldSize();
  bool is_master = (cur_rank == 0);

  static std::shared_ptr<TCPStore> store =
      std::make_shared<TCPStore>(host, port, is_master, world_size);
  return store;
}

}  // namespace phi::distributed
