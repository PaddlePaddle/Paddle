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

#pragma once

#include <map>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"

namespace cinn::adt {

template <typename NodeTypeT>
class UnionFind final {
 public:
  UnionFind() {}
  UnionFind(const UnionFind&) = delete;
  UnionFind(UnionFind&&) = delete;
  UnionFind& operator=(const UnionFind&) = delete;
  UnionFind& operator=(UnionFind&&) = delete;

  int Find(const NodeTypeT& input) {
    if (node2index_.find(input) == node2index_.end()) {
      return -1;
    }
    int idx = node2index_.at(input);
    while (idx != parents_.at(idx)) {
      int tmp = parents_.at(idx);
      parents_.at(idx) = parents_.at(tmp);
      idx = tmp;
    }
    return idx;
  }

  void Union(const NodeTypeT& lhs, const NodeTypeT& rhs) {
    int lhs_root = FindOrEmplace(lhs);
    int rhs_root = FindOrEmplace(rhs);
    if (lhs_root < rhs_root) {
      parents_[rhs_root] = lhs_root;
    } else if (lhs_root > rhs_root) {
      parents_[lhs_root] = rhs_root;
    } else {
      // Do nothing
    }
    return;
  }

  bool IsConnected(const NodeTypeT& lhs, const NodeTypeT& rhs) {
    return Find(lhs) != -1 && Find(rhs) != -1 && Find(lhs) == Find(rhs);
  }

  std::vector<std::vector<NodeTypeT>> AllNodeCluster() {
    std::map<int, std::vector<NodeTypeT>> root2cluster{};
    for (const auto& [node, _] : node2index_) {
      root2cluster[Find(node)].emplace_back(node);
    }
    std::vector<std::vector<NodeTypeT>> ret{};
    for (const auto& [_, cluster] : root2cluster) {
      ret.emplace_back(cluster);
    }
    return ret;
  }

  std::vector<NodeTypeT> NodeCluster(const NodeTypeT& node) {
    std::vector<NodeTypeT> ret{};
    for (const auto& [tmp_node, _] : node2index_) {
      if (IsConnected(tmp_node, node)) {
        ret.emplace_back(tmp_node);
      } else {
        // Do nothing
      }
    }
    return ret;
  }

 private:
  int FindOrEmplace(const NodeTypeT& node) {
    int root = Find(node);
    if (root == -1) {
      root = parents_.size();
      CHECK(node2index_.emplace(node, root).second);
      parents_.push_back(root);
    }
    return root;
  }

  // index -> parent_index
  std::vector<int> parents_;
  // NodeTypeT -> index
  std::unordered_map<NodeTypeT, int> node2index_;
};

}  // namespace cinn::adt
