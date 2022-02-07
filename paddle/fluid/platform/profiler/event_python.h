/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <map>

#include "paddle/fluid/platform/profiler/event_node.h"

namespace paddle {
namespace platform {

struct DevicePythonNode {
  DevicePythonNode() = default;
  ~DevicePythonNode() {}
  // record name
  std::string name;
  // record type, one of TracerEventType
  TracerEventType type;
  // start timestamp of the record
  uint64_t start_ns;
  // end timestamp of the record
  uint64_t end_ns;
  // device id
  uint64_t device_id;
  // context id
  uint64_t context_id;
  // stream id
  uint64_t stream_id;
};

struct HostPythonNode {
  HostPythonNode() = default;
  ~HostPythonNode();
  // record name
  std::string name;
  // record type, one of TracerEventType
  TracerEventType type;
  // start timestamp of the record
  uint64_t start_ns;
  // end timestamp of the record
  uint64_t end_ns;
  // process id of the record
  uint64_t process_id;
  // thread id of the record
  uint64_t thread_id;
  // children node
  std::vector<HostPythonNode*> children_node_ptrs;
  // runtime node
  std::vector<HostPythonNode*> runtime_node_ptrs;
  // device node
  std::vector<DevicePythonNode*> device_node_ptrs;
};

class ProfilerResult {
 public:
  ProfilerResult() : tree_(nullptr) {}
  explicit ProfilerResult(NodeTrees* tree);
  ~ProfilerResult();
  std::map<uint64_t, HostPythonNode*> GetData() {
    return thread_event_trees_map;
  }
  void Save(const std::string& file_name);

 private:
  std::map<uint64_t, HostPythonNode*> thread_event_trees_map;
  NodeTrees* tree_;
  HostPythonNode* CopyTree(HostTraceEventNode* node);
};

}  // namespace platform
}  // namespace paddle
