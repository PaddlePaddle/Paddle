/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/data_shard.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

std::future<void> DataShard::GetIndexsByIds(
    const std::vector<std::pair<int64_t, int64_t>>& id_to_offsets,
    std::vector<int64_t>* value_indexs, bool auto_grown) {
  auto task = [this, &id_to_offsets, value_indexs, auto_grown] {
    for (auto& id_to_offset : id_to_offsets) {
      int64_t id = id_to_offset.first;
      int64_t offset = id_to_offset.second;
      (*value_indexs)[offset] = GetIndexById(id, auto_grown);
    }
  };
  return pool_->enqueue(std::move(task));
}

inline int64_t DataShard::GetIndexById(int64_t id, bool auto_grown) {
  auto iter = id_to_offset_.find(id);
  if (iter == id_to_offset_.end()) {
    if (auto_grown) {
      auto shard_offset = id_to_offset_.size();
      PADDLE_ENFORCE_LT(shard_offset, shard_size_, "shard is full!");
      id_to_offset_[id] = shard_offset;
      return shard_id_ * shard_size_ + shard_offset;
    } else {
      return -1;
    }
  } else {
    return shard_id_ * shard_size_ + iter->second;
  }
}

}  // namespace framework
}  // namespace paddle
