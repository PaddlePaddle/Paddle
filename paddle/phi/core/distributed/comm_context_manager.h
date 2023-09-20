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

#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include "paddle/phi/core/distributed/comm_context.h"
#include "paddle/phi/core/macros.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/backends/gpu/forwards.h"
#endif

namespace phi {
namespace distributed {

class Store;

class CommContextManager {
 public:
  CommContextManager() = default;
  ~CommContextManager() = default;

  static CommContextManager& GetInstance() {
    static CommContextManager instance;
    return instance;
  }

  void SetStore(const std::shared_ptr<Store>& store) { store_ = store; }

  CommContext* Emplace(const std::string& unique_comm_key,
                       std::unique_ptr<CommContext> comm_context);

  CommContext* Get(const std::string& unique_comm_key) const;

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  int GetRingId(const ncclComm_t& comm) const;
#endif

  bool Has(const std::string& unique_comm_key) const;

  static void SetDeviceId(int dev_id);

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  static void CreateNCCLCommContext(const std::shared_ptr<Store>& store,
                                    const std::string& unique_comm_key,
                                    int rank,
                                    int size,
                                    const std::string& hash_key = "");
#endif

#if defined(PADDLE_WITH_GLOO)
  static void CreateGlooCommContext(const std::shared_ptr<Store>& store,
                                    const std::string& unique_comm_key,
                                    int rank,
                                    int size);
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
  static void CreateXCCLCommContext(const std::shared_ptr<Store>& store,
                                    const std::string& unique_comm_key,
                                    const std::string& device_type,
                                    int rank,
                                    int size);
#endif

 private:
  DISABLE_COPY_AND_ASSIGN(CommContextManager);

  std::unordered_map<std::string, std::unique_ptr<CommContext>>
      id_to_comm_context_;
  std::shared_ptr<Store> store_;
  static int device_id;
};

}  // namespace distributed
}  // namespace phi
