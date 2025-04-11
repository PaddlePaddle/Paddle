// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <unordered_map>
#include <vector>

namespace common {

template <typename T>
struct DefaultCompare {
  bool operator()(const T& a, const T& b) const { return a == b; }
};

template <typename T, typename Compare = DefaultCompare<T>>
class UnionFindSet {
 public:
  const T& Find(const T& x) const {
    if (parent_.find(x) == parent_.end()) {
      return x;
    }
    if (!compare_(parent_.at(x), x)) return Find(parent_.at(x));
    return parent_.at(x);
  }

  const T& Find(const T& x) {
    if (parent_.find(x) == parent_.end()) {
      return x;
    }
    if (!compare_(parent_.at(x), x)) {
      parent_[x] = Find(parent_[x]);
    }
    return parent_.at(x);
  }

  void Union(const T& p, const T& q) {
    if (parent_.find(p) == parent_.end()) {
      parent_[p] = p;
    }
    if (parent_.find(q) == parent_.end()) {
      parent_[q] = q;
    }
    parent_[Find(q)] = Find(p);
  }

  const std::unordered_map<T, T, std::hash<T>, Compare>& GetMap() const {
    return parent_;
  }

  template <typename DoEachClusterT>
  void VisitCluster(const DoEachClusterT& DoEachCluster) const {
    std::unordered_map<T, std::vector<T>> clusters_map;
    for (auto it = parent_.begin(); it != parent_.end(); it++) {
      clusters_map[Find(it->first)].emplace_back(it->first);
    }
    for (const auto& [_, clusters] : clusters_map) {
      DoEachCluster(clusters);
    }
  }

  bool HasSameRoot(const T& p, const T& q) const {
    // add shortcut for empty map.
    if (parent_.empty()) {
      return compare_(p, q);
    }
    return compare_(Find(p), Find(q));
  }

  std::unordered_map<T, T, std::hash<T>, Compare>* MutMap() { return &parent_; }

 private:
  std::unordered_map<T, T, std::hash<T>, Compare> parent_;
  Compare compare_;
};

}  // namespace common
