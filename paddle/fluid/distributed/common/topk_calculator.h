// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <queue>
#include <unordered_map>

namespace paddle {
namespace distributed {
class TopkCalculator {
 public:
  TopkCalculator(int shard_num, size_t k)
      : _shard_num(shard_num), _total_max_size(k) {
    _shard_max_size = _total_max_size / shard_num;
    _shard_max_size = _shard_max_size > 1 ? _shard_max_size : 1;
    for (int i = 0; i < shard_num; ++i) {
      _mpq.emplace(i, std::priority_queue<double, std::vector<double>,
                                          std::greater<double>>());
    }
  }
  ~TopkCalculator() {}
  bool push(int shard_id, double value) {
    if (_mpq.find(shard_id) == _mpq.end()) {
      return false;
    }
    auto &pq = _mpq[shard_id];
    if (pq.size() < _shard_max_size) {
      pq.push(value);
    } else {
      if (pq.top() < value) {
        pq.pop();
        pq.push(value);
      }
    }
    return true;
  }
  // TODO 再进行一次堆排序merge各个shard的结果
  int top() {
    double total = 0;
    for (const auto &item : _mpq) {
      auto &pq = item.second;
      if (!pq.empty()) {
        total += pq.top();
      }
    }
    return total / _shard_num;
  }

 private:
  std::unordered_map<int, std::priority_queue<double, std::vector<double>,
                                              std::greater<double>>>
      _mpq;
  int _shard_num;
  size_t _total_max_size;
  size_t _shard_max_size;
};

}  // namespace distributed
}  // namespace paddle
