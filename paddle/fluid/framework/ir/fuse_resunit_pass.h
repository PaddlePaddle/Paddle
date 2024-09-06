// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
// Copyright (c) 2023 NVIDIA Authors. All Rights Reserved.
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

#include <mutex>
#include <string>
#include <unordered_set>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

class Graph;
class Node;

class ResUnitPassCache {
 public:
  ResUnitPassCache() {}

  ResUnitPassCache(const ResUnitPassCache &) = delete;
  void operator=(const ResUnitPassCache &) = delete;

  bool Exists(const std::string &key) const { return var_map_.count(key); }

  ir::Node *Get(const std::string &key) {
    if (Exists(key)) {
      return var_map_.find(key)->second;
    }
    return nullptr;
  }

  void Insert(const std::string &key, ir::Node *const value) {
    if (!Exists(key)) {
      mtx.lock();
      var_map_.insert({key, value});
      mtx.unlock();
    } else {
      PADDLE_THROW(common::errors::AlreadyExists(
          "The key (%d) of ResUnitPassCache already exist.", key));
    }
  }

 private:
  std::unordered_map<std::string, ir::Node *> var_map_;
  std::mutex mtx;
};

class FuseResUnitPass : public FusePassBase {
 public:
  virtual ~FuseResUnitPass() {}

 protected:
  void ApplyImpl(ir::Graph *graph) const override;

  ir::Graph *FuseConvBNAddActFwd(
      ir::Graph *graph,
      const std::unordered_set<std::string> &act_types,
      bool shortcut,
      bool is_training) const;

  ir::Graph *FuseConvBNActConvBNstats(
      ir::Graph *graph,
      const std::unordered_set<std::string> &act_types,
      bool is_training,
      int *found_pattern_count_output,
      ResUnitPassCache *cache) const;

  ir::Graph *FuseBNActConvBwd(
      ir::Graph *graph,
      const std::unordered_set<std::string> &act_grad_types,
      ResUnitPassCache *cache) const;

  ir::Graph *FuseBNAddActConvBwd(
      ir::Graph *graph,
      const std::unordered_set<std::string> &act_grad_types,
      bool shortcut,
      bool with_sum) const;

 private:
  const std::string GetCacheKey(const std::string var_name,
                                int block_id) const {
    return std::to_string(block_id) + var_name;
  }
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
