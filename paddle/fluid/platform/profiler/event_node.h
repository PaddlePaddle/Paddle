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

#pragma once

#include <functional>
#include <list>
#include <map>
#include <string>
#include <vector>

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/profiler/event_record.h"
#include "paddle/fluid/platform/profiler/output_logger.h"

namespace paddle {
namespace platform {
class BaseLogger;  // forward declaration

class DeviceEventNode {
 public:
  // constructor
  explicit DeviceEventNode(const DeviceEvent& device_event)
      : device_event_(device_event) {}
  // destructor
  ~DeviceEventNode() {}
  // getter
  std::string name() const { return device_event_.name; }
  TracerEventType type() const { return device_event_.type; }
  uint64_t start_ns() const { return device_event_.start_ns; }
  uint64_t end_ns() const { return device_event_.end_ns; }
  uint64_t device_id() const { return device_event_.device_id; }
  uint64_t context_id() const { return device_event_.context_id; }
  uint64_t stream_id() const { return device_event_.stream_id; }
  uint64_t duration() const {
    return device_event_.end_ns - device_event_.start_ns;
  }
  uint32_t correlation_id() const { return device_event_.correlation_id; }
  KernelEventInfo kernel_info() const {
    PADDLE_ENFORCE_EQ(
        device_event_.type, TracerEventType::Kernel,
        platform::errors::Unavailable(
            "to get kernel_info, "
            "TracerEventType in node must be TracerEventType::Kernel"));
    return device_event_.kernel_info;
  }
  MemcpyEventInfo memcpy_info() const {
    PADDLE_ENFORCE_EQ(
        device_event_.type, TracerEventType::Memcpy,
        platform::errors::Unavailable(
            "to get memcpy_info, "
            "TracerEventType in node must be TracerEventType::Memcpy"));
    return device_event_.memcpy_info;
  }
  MemsetEventInfo memset_info() const {
    PADDLE_ENFORCE_EQ(
        device_event_.type, TracerEventType::Memset,
        platform::errors::Unavailable(
            "to get memset_info, "
            "TracerEventType in node must be TracerEventType::Memset"));
    return device_event_.memset_info;
  }

  // member function
  void LogMe(BaseLogger* logger) { logger->LogDeviceEventNode(*this); }

 private:
  // data
  DeviceEvent device_event_;
};

class CudaRuntimeEventNode {
 public:
  // constructor
  explicit CudaRuntimeEventNode(const RuntimeEvent& runtime_event)
      : runtime_event_(runtime_event) {}
  // destructor
  ~CudaRuntimeEventNode();
  // getter
  std::string name() const { return runtime_event_.name; }
  TracerEventType type() const { return runtime_event_.type; }
  uint64_t start_ns() const { return runtime_event_.start_ns; }
  uint64_t end_ns() const { return runtime_event_.end_ns; }
  uint64_t process_id() const { return runtime_event_.process_id; }
  uint64_t thread_id() const { return runtime_event_.thread_id; }
  uint64_t duration() const {
    return runtime_event_.end_ns - runtime_event_.start_ns;
  }
  uint32_t correlation_id() const { return runtime_event_.correlation_id; }
  uint32_t callback_id() const { return runtime_event_.callback_id; }
  // member function
  void AddDeviceEventNode(DeviceEventNode* node) {
    device_node_ptrs_.push_back(node);
  }
  void LogMe(BaseLogger* logger) { logger->LogRuntimeEventNode(*this); }
  std::vector<DeviceEventNode*>& GetDeviceEventNodes() {
    return device_node_ptrs_;
  }

 private:
  // data
  RuntimeEvent runtime_event_;
  // device events called by this
  std::vector<DeviceEventNode*> device_node_ptrs_;
};

class HostEventNode {
 public:
  // constructor
  explicit HostEventNode(const HostEvent& host_event)
      : host_event_(host_event) {}

  // destructor
  ~HostEventNode();

  // getter
  std::string name() const { return host_event_.name; }
  TracerEventType type() const { return host_event_.type; }
  uint64_t start_ns() const { return host_event_.start_ns; }
  uint64_t end_ns() const { return host_event_.end_ns; }
  uint64_t process_id() const { return host_event_.process_id; }
  uint64_t thread_id() const { return host_event_.thread_id; }
  uint64_t duration() const {
    return host_event_.end_ns - host_event_.start_ns;
  }

  // member function
  void AddChild(HostEventNode* node) { children_.push_back(node); }
  void AddCudaRuntimeNode(CudaRuntimeEventNode* node) {
    runtime_node_ptrs_.push_back(node);
  }
  std::vector<HostEventNode*>& GetChildren() { return children_; }
  std::vector<CudaRuntimeEventNode*>& GetRuntimeEventNodes() {
    return runtime_node_ptrs_;
  }
  void LogMe(BaseLogger* logger) { logger->LogHostEventNode(*this); }

 private:
  // data
  HostEvent host_event_;
  // cuda runtime events called by this
  std::vector<CudaRuntimeEventNode*> runtime_node_ptrs_;
  // host events called by this
  std::vector<HostEventNode*> children_;
};

class NodeTrees {
 public:
  // constructor
  NodeTrees(const std::list<HostEvent>& host_events,
            const std::list<RuntimeEvent>& runtime_events,
            const std::list<DeviceEvent>& device_events) {
    // encapsulate event into nodes
    for (auto it = host_events.begin(); it != host_events.end(); ++it) {
      host_event_nodes_.push_back(new HostEventNode(*it));
    }
    for (auto it = runtime_events.begin(); it != runtime_events.end(); ++it) {
      runtime_event_nodes_.push_back(new CudaRuntimeEventNode(*it));
    }
    for (auto it = device_events.begin(); it != device_events.end(); ++it) {
      device_event_nodes_.push_back(new DeviceEventNode(*it));
    }
    // build tree
    BuildTrees();
  }

  // destructor
  ~NodeTrees();

  void LogMe(BaseLogger* logger);
  void HandleTrees(std::function<void(HostEventNode*)>,
                   std::function<void(CudaRuntimeEventNode*)>,
                   std::function<void(DeviceEventNode*)>);
  std::map<uint64_t, HostEventNode*> GetNodeTrees() {
    return thread_event_trees_map_;
  }
  std::map<uint64_t, std::vector<HostEventNode*>> Traverse(bool bfs);

 private:
  std::map<uint64_t, HostEventNode*> thread_event_trees_map_;
  std::vector<HostEventNode*> host_event_nodes_;
  std::vector<CudaRuntimeEventNode*> runtime_event_nodes_;
  std::vector<DeviceEventNode*> device_event_nodes_;
  void BuildTrees();
  HostEventNode* BuildTreeRelationship(
      std::vector<HostEventNode*> host_event_nodes,
      std::vector<CudaRuntimeEventNode*> runtime_event_nodes);
};

}  // namespace platform
}  // namespace paddle
