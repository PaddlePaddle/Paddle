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

void DataShard::GetIndexsByIds(
    const std::vector<std::pair<int64_t, int64_t>>& id_to_offsets,
    std::vector<int64_t>* value_indexs, bool auto_grown, bool is_test) {
  for (auto& id_to_offset : id_to_offsets) {
    int64_t id = id_to_offset.first;
    int64_t offset = id_to_offset.second;
    int64_t index = GetIndexById(id, auto_grown, is_test);
    (*value_indexs)[offset] = index;
  }
}

int64_t DataShard::GetIndexById(int64_t id, bool auto_grown, bool is_test) {
  if (is_test) {
    auto iter = id_to_offset_.find(id);
    if (iter == id_to_offset_.end()) {
      return -1;
    } else {
      return shard_id_ * shard_size_ + iter->second;
    }
  }
  rwlock_->RDLock();
  auto iter = id_to_offset_.find(id);
  if (iter == id_to_offset_.end()) {
    rwlock_->UNLock();
    if (auto_grown) {
      rwlock_->WRLock();
      auto shard_offset = id_to_offset_.size();
      PADDLE_ENFORCE_LT(shard_offset, shard_size_, "shard is full!");
      id_to_offset_[id] = shard_offset;
      int64_t offset = shard_id_ * shard_size_ + shard_offset;
      rwlock_->UNLock();
      return offset;
    } else {
      return -1;
    }
  }
  int64_t offset = shard_id_ * shard_size_ + iter->second;
  rwlock_->UNLock();
  return offset;
}

void DataShard::GetAllIdToAbsOffset(std::vector<int64_t>& row_ids, std::vector<int64_t>& row_indexs) {
  rwlock_->RDLock();
  for (auto& iter : id_to_offset_) {
    row_ids.emplace_back(iter.first);
    row_indexs.emplace_back(shard_id_ * shard_size_ + iter.second);
  }
  rwlock_->UNLock();
  return;
}

void DataShard::ReconstructShardIndex(
    const std::unordered_map<int64_t, int64_t>& id_to_abs_offset) {
  rwlock_->WRLock();
  id_to_offset_.clear();
  for (auto& iter : id_to_abs_offset) {
    auto shard_offset = iter.second - (shard_id_ * shard_size_);
    id_to_offset_[iter.first] = shard_offset;
  }
  rwlock_->UNLock();
}

}  // namespace framework
}  // namespace paddle
