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

#pragma once

#include <algorithm>
#include <condition_variable>  // NOLINT
#include <mutex>               // NOLINT
#include <set>

#include "paddle/fluid/distributed/common/utils.h"
#include "paddle/fluid/distributed/ps/table/table.h"

namespace paddle {
namespace distributed {

template <typename T>
struct ReservoirValue {
  std::vector<T> values;
  uint32_t counter;
  uint32_t dim;

  ReservoirValue() {
    dim = 0;
    values.resize(dim);
    counter = 0;
  }

  explicit ReservoirValue(uint32_t dim) {
    this->dim = dim;
    values.resize(dim);
    counter = 0;
  }

  void add(const T *value, int numel) {
    GetBlas<T>().VADD(numel, values.data(), value, values.data());
    counter++;
  }

  void add(T *value, int numel) {
    GetBlas<T>().VADD(numel, values.data(), value, values.data());
    counter++;
  }

  void avg() {
    if (counter == 0) return;
    auto scale = 1 / static_cast<T>(counter);
    GetBlas<T>().SCAL(values.size(), scale, values.data());
  }

  void reset() {
    std::fill(values.begin(), values.end(), 0);
    counter = 0;
  }
};

class BarrierTable : public Table {
 public:
  BarrierTable() {}
  virtual ~BarrierTable() {}

  virtual void *GetShard(size_t shard_idx UNUSED) { return 0; }

  virtual int32_t Pull(TableContext &context UNUSED) { return 0; }  // NOLINT
  virtual int32_t Push(TableContext &context UNUSED) { return 0; }  // NOLINT

  int32_t Shrink(const std::string &param UNUSED) override { return 0; }
  virtual void Clear() {}
  virtual int32_t Flush() { return 0; }
  virtual int32_t Load(const std::string &path UNUSED,
                       const std::string &param UNUSED) {
    return 0;
  }
  virtual int32_t Save(const std::string &path UNUSED,
                       const std::string &param UNUSED) {
    return 0;
  }
#if defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_PSCORE)
  virtual int32_t Save_v2(const std::string &path, const std::string &param) {
    return 0;
  }
#endif
  virtual int32_t InitializeShard() { return 0; }

  int32_t Initialize() override;
  // only for barrier
  // 0: send_barrier 1: recv_barrier 2: complete
  int32_t Barrier(const uint32_t trainer_id,
                  const std::string barrier_type) override;

  int32_t SetTableMap(
      std::unordered_map<uint32_t, std::shared_ptr<Table>> *table_map) override;

 private:
  std::mutex mutex_;
  std::condition_variable trainer_wait_;
  std::set<uint64_t> trainer_ids_;
  std::set<uint64_t> trainer_all_;
  std::atomic<int> trigger_;
  std::atomic<bool> exit_;
  std::unordered_map<uint32_t, std::shared_ptr<Table>> *table_map_;
};
}  // namespace distributed
}  // namespace paddle
