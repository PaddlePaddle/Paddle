// Copyright (c) 2023 Enflame Authors. All Rights Reserved.
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

#include "paddle/fluid/platform/device/gcu/common/gcu_options.h"

#include <utility>

namespace paddle {
namespace platform {
namespace gcu {
GcuOptions &GetGcuOptions() {
  static thread_local GcuOptions thread_options;
  return thread_options;
}

std::string GcuOptions::GetOption(const std::string &key) {
  const std::map<std::string, std::string>::const_iterator graph_iter =
      graph_options_.find(key);
  if (graph_iter != graph_options_.end()) {
    return graph_iter->second;
  }
  const std::map<std::string, std::string>::const_iterator global_iter =
      global_options_.find(key);
  if (global_iter != global_options_.end()) {
    return global_iter->second;
  }
  return "";
}

void GcuOptions::SetGraphOption(const std::string &key,
                                const std::string &option) {
  graph_options_[key] = option;
}

void GcuOptions::SetGlobalOption(const std::string &key,
                                 const std::string &option) {
  global_options_[key] = option;
}

void GcuOptions::ResetGraphOptions(
    std::map<std::string, std::string> options_map) {
  graph_options_.clear();
  graph_options_ = std::move(options_map);
}

void GcuOptions::ResetGlobalOptions(
    std::map<std::string, std::string> options_map) {
  global_options_.clear();
  global_options_ = std::move(options_map);
}

std::map<std::string, std::string> GcuOptions::GetAllGraphOptions() const {
  return graph_options_;
}

std::map<std::string, std::string> GcuOptions::GetAllOptions() const {
  std::map<std::string, std::string> options_all;
  (void)options_all.insert(graph_options_.begin(), graph_options_.end());
  (void)options_all.insert(global_options_.begin(), global_options_.end());
  return options_all;
}

void GcuOptions::ClearGraphOption(const std::string &key) {
  (void)graph_options_.erase(key);
}

void GcuOptions::ClearGlobalOption(const std::string &key) {
  (void)global_options_.erase(key);
}

void GcuOptions::ClearAllOptions() {
  graph_options_.clear();
  global_options_.clear();
}

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
