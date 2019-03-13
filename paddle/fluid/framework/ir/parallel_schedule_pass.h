// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {
namespace ir {

const char kParallelMeta[] = "__parallel_meta__";

class ParallelMeta {
 public:
  // Only analyze the main block, so the key is string.
  // Each stream will have an event.
  using stream_map_t =
      std::unordered_map<std::string /*node repr*/, int /*stream id*/>;
  // Record the input events each operator depends on, and record the events the
  // outputs will update.
  using event_depend_map_t =
      std::unordered_map<std::string, std::set<int> /*stream ids*/>;

  void SetStreamId(const std::string& key, int id) {
    LOG(INFO) << "set stream " << key << ": " << id;
    stream_map_[key] = id;
  }
  int GetStreamId(const std::string& key) const {
    auto it = stream_map_.find(key);
    PADDLE_ENFORCE(it != stream_map_.end(), "no key %s in map", key);
    return it->second;
  }

  template <typename IterType>
  void SetInputDependEventIds(const std::string& key, IterType begin,
                              IterType end) {
    event_depend_map_[key + ":inputs"].insert(begin, end);
  }

  template <typename IterType>
  void SetOutputDependEventIds(const std::string& key, IterType begin,
                               IterType end) {
    event_depend_map_[key + ":outputs"].insert(begin, end);
  }

  const std::set<int>& GetInputDependEventIds(const std::string& key) const {
    auto it = event_depend_map_.find(key + ":inputs");
    PADDLE_ENFORCE(it != event_depend_map_.end(), "no key %s in map", key);
    return it->second;
  }

  const std::set<int>& GetOutputDependEventIds(const std::string& key) const {
    auto it = event_depend_map_.find(key + ":outputs");
    PADDLE_ENFORCE(it != event_depend_map_.end(), "no key %s in map", key);
    return it->second;
  }

  std::set<int> StreamIds() const {
    std::set<int> set;
    for (auto item : stream_map_) {
      set.insert(item.second);
    }
    return set;
  }

 private:
  stream_map_t stream_map_;
  event_depend_map_t event_depend_map_;
};

class ParallelSchedulePass : public Pass {
 public:
 protected:
  std::unique_ptr<Graph> ApplyImpl(std::unique_ptr<Graph> graph) const override;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
