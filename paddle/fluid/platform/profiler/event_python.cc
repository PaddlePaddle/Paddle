/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/profiler/event_python.h"
#include "paddle/fluid/platform/profiler/output_logger.h"

namespace paddle {
namespace platform {

HostPythonNode::~HostPythonNode() {
  // delete all runtime or device nodes and recursive delete children
  for (auto it = children_node_ptrs.begin(); it != children_node_ptrs.end();
       ++it) {
    delete *it;
  }
  for (auto it = runtime_node_ptrs.begin(); it != runtime_node_ptrs.end();
       ++it) {
    delete *it;
  }
  for (auto it = device_node_ptrs.begin(); it != device_node_ptrs.end(); ++it) {
    delete *it;
  }
}

HostPythonNode* ProfilerResult::_copyTree(HostEventNode* node) {
  // Copy and transfer EventNode in NodeTree to PythonNode
  if (node == nullptr) {
    return nullptr;
  }
  // copy HostEventNode and its children
  HostPythonNode* host_python_node = new HostPythonNode();
  host_python_node->name = node->name();
  host_python_node->type = node->type();
  host_python_node->start_ns = node->start_ns();
  host_python_node->end_ns = node->end_ns();
  host_python_node->process_id = node->process_id();
  host_python_node->thread_id = node->thread_id();
  for (auto it = node->GetChildren().begin(); it != node->GetChildren().end();
       ++it) {
    host_python_node->children_node_ptrs.push_back(_copyTree(*it));
  }
  // copy its CudaRuntimeEventNode
  for (auto runtimenode = node->GetRuntimeEventNodes().begin();
       runtimenode != node->GetRuntimeEventNodes().end(); ++runtimenode) {
    HostPythonNode* runtime_python_node = new HostPythonNode();
    runtime_python_node->name = (*runtimenode)->name();
    runtime_python_node->type = (*runtimenode)->type();
    runtime_python_node->start_ns = (*runtimenode)->start_ns();
    runtime_python_node->end_ns = (*runtimenode)->end_ns();
    runtime_python_node->process_id = (*runtimenode)->process_id();
    runtime_python_node->thread_id = (*runtimenode)->thread_id();
    host_python_node->runtime_node_ptrs.push_back(runtime_python_node);
    // copy DeviceEventNode
    for (auto devicenode = (*runtimenode)->GetDeviceEventNodes().begin();
         devicenode != (*runtimenode)->GetDeviceEventNodes().end();
         ++devicenode) {
      DevicePythonNode* device_python_node = new DevicePythonNode();
      device_python_node->name = (*devicenode)->name();
      device_python_node->type = (*devicenode)->type();
      device_python_node->start_ns = (*devicenode)->start_ns();
      device_python_node->end_ns = (*devicenode)->end_ns();
      device_python_node->device_id = (*devicenode)->device_id();
      device_python_node->context_id = (*devicenode)->context_id();
      device_python_node->stream_id = (*devicenode)->stream_id();
      runtime_python_node->device_node_ptrs.push_back(device_python_node);
    }
  }
  return host_python_node;
}

ProfilerResult::ProfilerResult(NodeTrees* tree) : tree_(tree) {
  if (tree_ != nullptr) {
    std::map<uint64_t, HostEventNode*> nodetrees = tree_->GetNodeTrees();
    for (auto it = nodetrees.begin(); it != nodetrees.end(); ++it) {
      thread_event_trees_map[it->first] = _copyTree(it->second);
    }
  }
}

ProfilerResult::~ProfilerResult() {
  // delete all root nodes
  for (auto it = thread_event_trees_map.begin();
       it != thread_event_trees_map.end(); ++it) {
    delete it->second;
  }
}

void ProfilerResult::Save(const std::string& file_name) {
  ChromeTracingLogger logger(file_name);
  tree_->LogMe(&logger);
  return;
}

}  // namespace platform
}  // namespace paddle
