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

#include <functional>
#include <list>
#include <map>
#include <string>
#include <vector>

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/profiler/trace_event.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/platform/profiler/output_logger.h"

namespace paddle {
namespace platform {

class MemTraceEventNode {
 public:
  // constructor
  explicit MemTraceEventNode(const MemTraceEvent& mem_event)
      : mem_event_(mem_event) {}

  // destructor
  ~MemTraceEventNode();

  // getter
  TracerMemEventType Type() const { return mem_event_.type; }
  uint64_t Addr() const { return mem_event_.addr; }
  uint64_t TimeStampNs() const { return mem_event_.timestamp_ns; }
  uint64_t ProcessId() const { return mem_event_.process_id; }
  uint64_t ThreadId() const { return mem_event_.thread_id; }
  int64_t IncreaseBytes() const { return mem_event_.increase_bytes; }
  std::string Place() const { return mem_event_.place; }
  uint64_t CurrentAllocated() const { return mem_event_.current_allocated; }
  uint64_t CurrentReserved() const { return mem_event_.current_reserved; }
  uint64_t PeakAllocated() const { return mem_event_.peak_allocated; }
  uint64_t PeakReserved() const { return mem_event_.peak_reserved; }

  // member function
  void LogMe(BaseLogger* logger) { logger->LogMemTraceEventNode(*this); }

 private:
  // data
  MemTraceEvent mem_event_;
};

class OperatorSupplementEventNode {
 public:
  // constructor
  explicit OperatorSupplementEventNode(
      const OperatorSupplementEvent& op_supplement_event)
      : op_supplement_event_(op_supplement_event) {}
  // destructor
  ~OperatorSupplementEventNode() {}
  // getter
  std::string Name() const { return op_supplement_event_.op_type; }
  uint64_t TimeStampNs() const { return op_supplement_event_.timestamp_ns; }
  std::map<std::string, std::vector<std::vector<int64_t>>>& InputShapes() {
    return op_supplement_event_.input_shapes;
  }
  std::map<std::string, std::vector<std::string>>& Dtypes() {
    return op_supplement_event_.dtypes;
  }
  std::string CallStack() { return op_supplement_event_.callstack; }
  framework::AttributeMap Attributes() {
    return op_supplement_event_.attributes;
  }
  uint64_t OpId() const { return op_supplement_event_.op_id; }
  uint64_t ProcessId() const { return op_supplement_event_.process_id; }
  uint64_t ThreadId() const { return op_supplement_event_.thread_id; }

 private:
  // data
  OperatorSupplementEvent op_supplement_event_;
};

class DeviceTraceEventNode {
 public:
  // constructor
  explicit DeviceTraceEventNode(const DeviceTraceEvent& device_event)
      : device_event_(device_event) {}
  // destructor
  ~DeviceTraceEventNode() {}
  // getter
  std::string Name() const { return device_event_.name; }
  TracerEventType Type() const { return device_event_.type; }
  uint64_t StartNs() const { return device_event_.start_ns; }
  uint64_t EndNs() const { return device_event_.end_ns; }
  uint64_t DeviceId() const { return device_event_.device_id; }
  uint64_t ContextId() const { return device_event_.context_id; }
  uint64_t StreamId() const { return device_event_.stream_id; }
  uint64_t Duration() const {
    return device_event_.end_ns - device_event_.start_ns;
  }
  uint32_t CorrelationId() const { return device_event_.correlation_id; }
  KernelEventInfo KernelInfo() const {
    PADDLE_ENFORCE_EQ(
        device_event_.type,
        TracerEventType::Kernel,
        common::errors::Unavailable(
            "Can not kernel_info, "
            "TracerEventType in node must be TracerEventType::Kernel."));
    return device_event_.kernel_info;
  }
  MemcpyEventInfo MemcpyInfo() const {
    PADDLE_ENFORCE_EQ(
        device_event_.type,
        TracerEventType::Memcpy,
        common::errors::Unavailable(
            "Can not get memcpy_info, "
            "TracerEventType in node must be TracerEventType::Memcpy."));
    return device_event_.memcpy_info;
  }
  MemsetEventInfo MemsetInfo() const {
    PADDLE_ENFORCE_EQ(
        device_event_.type,
        TracerEventType::Memset,
        common::errors::Unavailable(
            "Can not get memset_info, "
            "TracerEventType in node must be TracerEventType::Memset."));
    return device_event_.memset_info;
  }

  // member function
  void LogMe(BaseLogger* logger) { logger->LogDeviceTraceEventNode(*this); }

 private:
  // data
  DeviceTraceEvent device_event_;
};

class CudaRuntimeTraceEventNode {
 public:
  // constructor
  explicit CudaRuntimeTraceEventNode(const RuntimeTraceEvent& runtime_event)
      : runtime_event_(runtime_event) {}
  // destructor
  ~CudaRuntimeTraceEventNode();
  // getter
  std::string Name() const { return runtime_event_.name; }
  TracerEventType Type() const { return runtime_event_.type; }
  uint64_t StartNs() const { return runtime_event_.start_ns; }
  uint64_t EndNs() const { return runtime_event_.end_ns; }
  uint64_t ProcessId() const { return runtime_event_.process_id; }
  uint64_t ThreadId() const { return runtime_event_.thread_id; }
  uint64_t Duration() const {
    return runtime_event_.end_ns - runtime_event_.start_ns;
  }
  uint32_t CorrelationId() const { return runtime_event_.correlation_id; }
  uint32_t CallbackId() const { return runtime_event_.callback_id; }
  // member function
  void AddDeviceTraceEventNode(DeviceTraceEventNode* node) {
    device_node_ptrs_.push_back(node);
  }
  void LogMe(BaseLogger* logger) { logger->LogRuntimeTraceEventNode(*this); }
  const std::vector<DeviceTraceEventNode*>& GetDeviceTraceEventNodes() const {
    return device_node_ptrs_;
  }

 private:
  // data
  RuntimeTraceEvent runtime_event_;
  // device events called by this
  std::vector<DeviceTraceEventNode*> device_node_ptrs_;
};

class HostTraceEventNode {
 public:
  // constructor
  explicit HostTraceEventNode(const HostTraceEvent& host_event)
      : host_event_(host_event) {}

  // destructor
  ~HostTraceEventNode();

  // getter
  std::string Name() const { return host_event_.name; }
  TracerEventType Type() const { return host_event_.type; }
  uint64_t StartNs() const { return host_event_.start_ns; }
  uint64_t EndNs() const { return host_event_.end_ns; }
  uint64_t ProcessId() const { return host_event_.process_id; }
  uint64_t ThreadId() const { return host_event_.thread_id; }
  uint64_t Duration() const {
    return host_event_.end_ns - host_event_.start_ns;
  }

  // member function
  void AddChild(HostTraceEventNode* node) { children_.push_back(node); }
  void AddCudaRuntimeNode(CudaRuntimeTraceEventNode* node) {
    runtime_node_ptrs_.push_back(node);
  }
  void AddMemNode(MemTraceEventNode* node) { mem_node_ptrs_.push_back(node); }
  void SetOperatorSupplementNode(OperatorSupplementEventNode* node) {
    op_supplement_node_ptr_ = node;
  }
  const std::vector<HostTraceEventNode*>& GetChildren() const {
    return children_;
  }
  const std::vector<CudaRuntimeTraceEventNode*>& GetRuntimeTraceEventNodes()
      const {
    return runtime_node_ptrs_;
  }
  const std::vector<MemTraceEventNode*>& GetMemTraceEventNodes() const {
    return mem_node_ptrs_;
  }

  OperatorSupplementEventNode* GetOperatorSupplementEventNode() const {
    return op_supplement_node_ptr_;
  }

  void LogMe(BaseLogger* logger) { logger->LogHostTraceEventNode(*this); }

 private:
  // data
  HostTraceEvent host_event_;
  // cuda runtime events called by this
  std::vector<CudaRuntimeTraceEventNode*> runtime_node_ptrs_;
  // host events called by this
  std::vector<HostTraceEventNode*> children_;
  // memory events happened in this event period
  std::vector<MemTraceEventNode*> mem_node_ptrs_;
  OperatorSupplementEventNode* op_supplement_node_ptr_ = nullptr;
};

class NodeTrees {
 public:
  // constructor
  NodeTrees(const std::list<HostTraceEvent>& host_events,
            const std::list<RuntimeTraceEvent>& runtime_events,
            const std::list<DeviceTraceEvent>& device_events,
            const std::list<MemTraceEvent>& mem_events,
            const std::list<OperatorSupplementEvent>& op_supplement_events) {
    std::vector<HostTraceEventNode*> host_event_nodes;
    std::vector<CudaRuntimeTraceEventNode*> runtime_event_nodes;
    std::vector<DeviceTraceEventNode*> device_event_nodes;
    std::vector<MemTraceEventNode*> mem_event_nodes;
    std::vector<OperatorSupplementEventNode*> op_supplement_event_nodes;
    // encapsulate event into nodes
    for (auto it = host_events.begin(); it != host_events.end(); ++it) {
      host_event_nodes.push_back(new HostTraceEventNode(*it));
    }
    for (auto it = runtime_events.begin(); it != runtime_events.end(); ++it) {
      runtime_event_nodes.push_back(new CudaRuntimeTraceEventNode(*it));
    }
    for (auto it = device_events.begin(); it != device_events.end(); ++it) {
      device_event_nodes.push_back(new DeviceTraceEventNode(*it));
    }
    for (auto it = mem_events.begin(); it != mem_events.end(); ++it) {
      mem_event_nodes.push_back(new MemTraceEventNode(*it));
    }
    for (auto it = op_supplement_events.begin();
         it != op_supplement_events.end();
         ++it) {
      op_supplement_event_nodes.push_back(new OperatorSupplementEventNode(*it));
    }
    // build tree
    BuildTrees(host_event_nodes,
               runtime_event_nodes,
               device_event_nodes,
               mem_event_nodes,
               op_supplement_event_nodes);
  }

  explicit NodeTrees(
      const std::map<uint64_t, HostTraceEventNode*>& thread_event_trees_map)
      : thread_event_trees_map_(thread_event_trees_map) {}

  // destructor
  ~NodeTrees();

  void LogMe(BaseLogger* logger);
  void HandleTrees(std::function<void(HostTraceEventNode*)>,
                   std::function<void(CudaRuntimeTraceEventNode*)>,
                   std::function<void(DeviceTraceEventNode*)>,
                   std::function<void(MemTraceEventNode*)>,
                   std::function<void(OperatorSupplementEventNode*)>);
  const std::map<uint64_t, HostTraceEventNode*>& GetNodeTrees() const {
    return thread_event_trees_map_;
  }
  std::map<uint64_t, std::vector<HostTraceEventNode*>> Traverse(bool bfs) const;

 private:
  std::map<uint64_t, HostTraceEventNode*> thread_event_trees_map_;
  void BuildTrees(const std::vector<HostTraceEventNode*>&,
                  const std::vector<CudaRuntimeTraceEventNode*>&,
                  const std::vector<DeviceTraceEventNode*>&,
                  const std::vector<MemTraceEventNode*>&,
                  const std::vector<OperatorSupplementEventNode*>&);
  HostTraceEventNode* BuildTreeRelationship(
      std::vector<HostTraceEventNode*> host_event_nodes,
      std::vector<CudaRuntimeTraceEventNode*> runtime_event_nodes,
      std::vector<MemTraceEventNode*> mem_event_nodes,
      std::vector<OperatorSupplementEventNode*> op_supplement_event_nodes);
};

}  // namespace platform
}  // namespace paddle
