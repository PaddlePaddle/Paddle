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

#pragma once

#include <future>  // NOLINT
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/rw_lock.h"

namespace paddle {
namespace framework {

/*
 * split selected rows into multiple shards, each shard will has an thread to
 * update it.
 */
class DataShard {
 public:
  DataShard(int64_t shard_id, int64_t shard_size)
      : shard_id_(shard_id),
        shard_size_(shard_size) {
    rwlock_.reset(new RWLock());
    }

  void GetIndexsByIds(
      const std::vector<std::pair<int64_t, int64_t>>& id_to_offsets,
      std::vector<int64_t>* value_indexs, bool auto_grown, bool is_test);

  int64_t GetIndexById(int64_t id, bool auto_grown, bool is_test);

  void GetAllIdToAbsOffset(std::vector<int64_t>& row_ids, std::vector<int64_t>& row_indexs);

  void ReconstructShardIndex(
      const std::unordered_map<int64_t, int64_t>& id_to_offset);

 private:
  std::unordered_map<int64_t, int64_t> id_to_offset_;
  int64_t shard_id_;
  int64_t shard_size_;
  std::unique_ptr<RWLock> rwlock_{nullptr};
};

}  // namespace framework
}  // namespace paddle
