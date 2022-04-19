// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/ps/table/common_table.h"

namespace paddle {
namespace distributed {

int32_t BarrierTable::Initialize() {
  auto trainers = _config.common().trainer_num();
  trigger_.store(trainers);

  for (int x = 0; x < trainers; ++x) {
    trainer_all_.insert(x);
  }
  VLOG(1) << "BarrierTable init trigger: " << trigger_.load();
  return 0;
}

// 0: send_barrier 1: recv_barrier 2: complete
int32_t BarrierTable::Barrier(const uint32_t trainer_id,
                              const std::string barrier_type) {
  std::unique_lock<std::mutex> lock(mutex_);

  if (barrier_type == "2") {
    trigger_.fetch_sub(1, std::memory_order::memory_order_relaxed);
    VLOG(1) << "trigger sub to : " << trigger_.load();
  } else {
    trainer_ids_.insert(trainer_id);
    VLOG(1) << "barrier type: " << barrier_type
            << " add trainer id: " << trainer_id;
  }

  if (trainer_ids_.size() < trigger_.load()) {
    std::vector<uint32_t> diffs(trainer_all_.size());
    auto iter = std::set_difference(trainer_all_.begin(), trainer_all_.end(),
                                    trainer_ids_.begin(), trainer_ids_.end(),
                                    diffs.begin());
    diffs.resize(iter - diffs.begin());

    auto diff = to_string<uint32_t>(diffs);
    VLOG(1) << "still need trainers: " << diff;
    trainer_wait_.wait(lock, [&] { return trainer_ids_.size() == 0; });
  } else {
    VLOG(1) << "barrier table optimize begin";
    for (auto& x : *table_map_) {
      auto table = x.second;
      table->Pour();
    }
    VLOG(1) << "barrier table optimize done";

    trainer_ids_.clear();
    trainer_wait_.notify_all();
  }
  return 0;
}

int32_t BarrierTable::SetTableMap(
    std::unordered_map<uint32_t, std::shared_ptr<Table>>* table_map) {
  table_map_ = table_map;
  return 0;
}

}  // namespace distributed
}  // namespace paddle
