// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

/**
 * \file This file implements a general UnionFind algorithm to help cluster
 * something.
 */
#pragma once
#include <cstring>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "paddle/cinn/common/object.h"
#include "paddle/cinn/common/shared.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace common {

struct UnionFindNode : public Object {
  UnionFindNode* parent{};
  std::string cluster_info;

  std::tuple<UnionFindNode*, int /*height*/> GetRoot() {
    auto* p = this;
    int level = 0;
    while (p->parent) {
      p = p->parent;
      level++;
    }
    return std::make_tuple(p, level);
  }

  void Union(UnionFindNode* other) {
    auto _p0_l0_ = GetRoot();
    auto& p0 = std::get<0>(_p0_l0_);
    auto& l0 = std::get<1>(_p0_l0_);
    auto _p1_l1_ = other->GetRoot();
    auto& p1 = std::get<0>(_p1_l1_);
    auto& l1 = std::get<1>(_p1_l1_);
    if (p0 == p1) return;

    if (l0 < l1) {
      p1->parent = p0;
    } else {
      p0->parent = p1;
    }
  }

  template <typename T>
  T* safe_as() {
    PADDLE_ENFORCE_EQ(
        std::strcmp(T::__type_info__, type_info()),
        0,
        ::common::errors::InvalidArgument(
            "Want a %d but get a %d", T::__type_info__, type_info()));
    return reinterpret_cast<T*>(this);
  }

  const char* type_info() const override;

  static const char* __type_info__;
};

struct UnionFind {
  UnionFindNode* AddNode(UnionFindNode* node) {
    nodes.emplace_back(node);
    return node;
  }

  std::vector<std::vector<UnionFindNode*>> GetClusters() {
    std::map<UnionFindNode* /*root*/, std::vector<UnionFindNode*>> clusters;

    for (auto& n : nodes) {
      auto _root_l_ = n->GetRoot();  // NOLINT
      auto& root = std::get<0>(_root_l_);
      auto& l = std::get<1>(_root_l_);
      clusters[root].push_back(n.get());
    }

    std::vector<std::vector<UnionFindNode*>> res;
    for (auto& item : clusters) {
      res.push_back(item.second);
    }
    return res;
  }

  std::vector<cinn::common::Shared<UnionFindNode>> nodes;
};

template <typename T>
class UnionFindSet {
 public:
  T Find(const T& x) {
    if (parent_.find(x) == parent_.end()) {
      return x;
    }
    if (parent_[x] != x) {
      parent_[x] = Find(parent_[x]);
    }
    return parent_[x];
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

  std::vector<std::vector<T>> Clusters() const {
    std::unordered_map<T, std::vector<T>> clusters_map;
    for (auto it = parent_.begin(); it != parent_.end(); it++) {
      clusters_map[it->second].emplace_back(it->first);
    }
    std::vector<std::vector<T>> clusters;
    for (auto it = clusters_map.begin(); it != clusters_map.end(); it++) {
      clusters.emplace_back(it->second);
    }
    return clusters;
  }

 private:
  std::unordered_map<T, T> parent_;
};

}  // namespace common
}  // namespace cinn
