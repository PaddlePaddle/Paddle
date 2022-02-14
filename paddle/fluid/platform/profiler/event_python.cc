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
#include "paddle/fluid/platform/profiler/chrometracing_logger.h"
#include "paddle/fluid/platform/profiler/dump/serialization_logger.h"

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

HostPythonNode* ProfilerResult::CopyTree(HostTraceEventNode* node) {
  // Copy and transfer EventNode in NodeTree to PythonNode
  if (node == nullptr) {
    return nullptr;
  }
  // copy HostTraceEventNode and its children
  HostPythonNode* host_python_node = new HostPythonNode();
  host_python_node->name = node->Name();
  host_python_node->type = node->Type();
  host_python_node->start_ns = node->StartNs();
  host_python_node->end_ns = node->EndNs();
  host_python_node->process_id = node->ProcessId();
  host_python_node->thread_id = node->ThreadId();
  for (auto it = node->GetChildren().begin(); it != node->GetChildren().end();
       ++it) {
    host_python_node->children_node_ptrs.push_back(CopyTree(*it));
  }
  // copy its CudaRuntimeTraceEventNode
  for (auto runtimenode = node->GetRuntimeTraceEventNodes().begin();
       runtimenode != node->GetRuntimeTraceEventNodes().end(); ++runtimenode) {
    HostPythonNode* runtime_python_node = new HostPythonNode();
    runtime_python_node->name = (*runtimenode)->Name();
    runtime_python_node->type = (*runtimenode)->Type();
    runtime_python_node->start_ns = (*runtimenode)->StartNs();
    runtime_python_node->end_ns = (*runtimenode)->EndNs();
    runtime_python_node->process_id = (*runtimenode)->ProcessId();
    runtime_python_node->thread_id = (*runtimenode)->ThreadId();
    host_python_node->runtime_node_ptrs.push_back(runtime_python_node);
    // copy DeviceTraceEventNode
    for (auto devicenode = (*runtimenode)->GetDeviceTraceEventNodes().begin();
         devicenode != (*runtimenode)->GetDeviceTraceEventNodes().end();
         ++devicenode) {
      DevicePythonNode* device_python_node = new DevicePythonNode();
      device_python_node->name = (*devicenode)->Name();
      device_python_node->type = (*devicenode)->Type();
      device_python_node->start_ns = (*devicenode)->StartNs();
      device_python_node->end_ns = (*devicenode)->EndNs();
      device_python_node->device_id = (*devicenode)->DeviceId();
      device_python_node->context_id = (*devicenode)->ContextId();
      device_python_node->stream_id = (*devicenode)->StreamId();
      runtime_python_node->device_node_ptrs.push_back(device_python_node);
    }
  }
  return host_python_node;
}

ProfilerResult::ProfilerResult(std::unique_ptr<NodeTrees> tree)
    : tree_(std::move(tree)) {
  if (tree_ != nullptr) {
    std::map<uint64_t, HostTraceEventNode*> nodetrees = tree_->GetNodeTrees();
    for (auto it = nodetrees.begin(); it != nodetrees.end(); ++it) {
      thread_event_trees_map[it->first] = CopyTree(it->second);
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

void ProfilerResult::Save(const std::string& file_name,
                          const std::string format) {
  if (format == std::string("json")) {
    ChromeTracingLogger logger(file_name);
    tree_->LogMe(&logger);
  } else if (format == std::string("pb")) {
    SerializationLogger logger(file_name);
    tree_->LogMe(&logger);
  }
  return;
}

}  // namespace platform
}  // namespace paddle
