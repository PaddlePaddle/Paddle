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
#include <unordered_map>

#include "paddle/phi/core/distributed/comm_context.h"
#include "paddle/phi/core/macros.h"

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

  CommContext* Emplace(int ring_id, std::unique_ptr<CommContext> comm_context);

  CommContext* Get(int ring_id) const;

  bool Has(int ring_id) const;

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  static void CreateNCCLCommContext(const std::shared_ptr<Store>& store,
                                    int dev_id,
                                    int ring_id,
                                    int rank,
                                    int size);
#endif

#if defined(PADDLE_WITH_GLOO)
  static void CreateGlooCommContext(const std::shared_ptr<Store>& store,
                                    int ring_id,
                                    int rank,
                                    int size);
#endif

 private:
  DISABLE_COPY_AND_ASSIGN(CommContextManager);

  std::unordered_map<int, std::unique_ptr<CommContext>> id_to_comm_context_;
  std::shared_ptr<Store> store_;
};

}  // namespace distributed
}  // namespace phi
