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

class DeviceRecordNode {
 public:
  // constructor
  explicit DeviceRecordNode(const DeviceRecord& device_record)
      : device_record_(device_record) {}
  // destructor
  ~DeviceRecordNode() {}
  // getter
  std::string name() const { return device_record_.name; }
  TracerEventType type() const { return device_record_.type; }
  uint64_t start_ns() const { return device_record_.start_ns; }
  uint64_t end_ns() const { return device_record_.end_ns; }
  uint64_t device_id() const { return device_record_.device_id; }
  uint64_t context_id() const { return device_record_.context_id; }
  uint64_t stream_id() const { return device_record_.stream_id; }
  uint64_t duration() const {
    return device_record_.end_ns - device_record_.start_ns;
  }
  uint32_t correlation_id() const { return device_record_.correlation_id; }
  KernelRecordInfo kernel_info() const {
    PADDLE_ENFORCE_EQ(
        device_record_.type, TracerEventType::Kernel,
        platform::errors::Unavailable(
            "to get kernel_info, "
            "TracerEventType in node must be TracerEventType::Kernel"));
    return device_record_.kernel_info;
  }
  MemcpyRecordInfo memcpy_info() const {
    PADDLE_ENFORCE_EQ(
        device_record_.type, TracerEventType::Memcpy,
        platform::errors::Unavailable(
            "to get memcpy_info, "
            "TracerEventType in node must be TracerEventType::Memcpy"));
    return device_record_.memcpy_info;
  }
  MemsetRecordInfo memset_info() const {
    PADDLE_ENFORCE_EQ(
        device_record_.type, TracerEventType::Memset,
        platform::errors::Unavailable(
            "to get memset_info, "
            "TracerEventType in node must be TracerEventType::Memset"));
    return device_record_.memset_info;
  }

  // member function
  void LogMe(BaseLogger* logger) { logger->LogDeviceRecordNode(*this); }

 private:
  // data
  DeviceRecord device_record_;
};

class CudaRuntimeRecordNode {
 public:
  // constructor
  explicit CudaRuntimeRecordNode(const RuntimeRecord& runtime_record)
      : runtime_record_(runtime_record) {}
  // destructor
  ~CudaRuntimeRecordNode();
  // getter
  std::string name() const { return runtime_record_.name; }
  TracerEventType type() const { return runtime_record_.type; }
  uint64_t start_ns() const { return runtime_record_.start_ns; }
  uint64_t end_ns() const { return runtime_record_.end_ns; }
  uint64_t process_id() const { return runtime_record_.process_id; }
  uint64_t thread_id() const { return runtime_record_.thread_id; }
  uint64_t duration() const {
    return runtime_record_.end_ns - runtime_record_.start_ns;
  }
  uint32_t correlation_id() const { return runtime_record_.correlation_id; }
  uint32_t callback_id() const { return runtime_record_.callback_id; }
  // member function
  void AddDeviceRecordNode(DeviceRecordNode* node) {
    device_node_ptrs_.push_back(node);
  }
  void LogMe(BaseLogger* logger) { logger->LogRuntimeRecordNode(*this); }
  std::vector<DeviceRecordNode*> GetDeviceRecordNodes() {
    return device_node_ptrs_;
  }

 private:
  // data
  RuntimeRecord runtime_record_;
  // device records called by this
  std::vector<DeviceRecordNode*> device_node_ptrs_;
};

class HostRecordNode {
 public:
  // constructor
  explicit HostRecordNode(const HostRecord& host_record)
      : host_record_(host_record) {}

  // destructor
  ~HostRecordNode();

  // getter
  std::string name() const { return host_record_.name; }
  TracerEventType type() const { return host_record_.type; }
  uint64_t start_ns() const { return host_record_.start_ns; }
  uint64_t end_ns() const { return host_record_.end_ns; }
  uint64_t process_id() const { return host_record_.process_id; }
  uint64_t thread_id() const { return host_record_.thread_id; }
  uint64_t duration() const {
    return host_record_.end_ns - host_record_.start_ns;
  }

  // member function
  void AddChild(HostRecordNode* node) { children_.push_back(node); }
  void AddCudaRuntimeNode(CudaRuntimeRecordNode* node) {
    runtime_node_ptrs_.push_back(node);
  }
  std::vector<HostRecordNode*>& GetChildren() { return children_; }
  std::vector<CudaRuntimeRecordNode*>& GetRuntimeRecordNodes() {
    return runtime_node_ptrs_;
  }
  void LogMe(BaseLogger* logger) { logger->LogHostRecordNode(*this); }

 private:
  // data
  HostRecord host_record_;
  // cuda runtime records called by this
  std::vector<CudaRuntimeRecordNode*> runtime_node_ptrs_;
  // host records called by this
  std::vector<HostRecordNode*> children_;
};

class NodeTrees {
 public:
  // constructor
  NodeTrees(const std::list<HostRecord>& host_records,
            const std::list<RuntimeRecord>& runtime_records,
            const std::list<DeviceRecord>& device_records) {
    // encapsulate record into nodes
    for (auto it = host_records.begin(); it != host_records.end(); ++it) {
      host_record_nodes_.push_back(new HostRecordNode(*it));
    }
    for (auto it = runtime_records.begin(); it != runtime_records.end(); ++it) {
      runtime_record_nodes_.push_back(new CudaRuntimeRecordNode(*it));
    }
    for (auto it = device_records.begin(); it != device_records.end(); ++it) {
      device_record_nodes_.push_back(new DeviceRecordNode(*it));
    }
    // build tree
    BuildTrees();
  }

  // destructor
  ~NodeTrees();

  void LogMe(BaseLogger* logger);
  void HandleTrees(std::function<void(HostRecordNode*)>,
                   std::function<void(CudaRuntimeRecordNode*)>,
                   std::function<void(DeviceRecordNode*)>);
  std::map<uint64_t, HostRecordNode*> GetNodeTrees() {
    return thread_record_trees_map_;
  }
  std::map<uint64_t, std::vector<HostRecordNode*>> Traverse(bool bfs);

 private:
  std::map<uint64_t, HostRecordNode*> thread_record_trees_map_;
  std::vector<HostRecordNode*> host_record_nodes_;
  std::vector<CudaRuntimeRecordNode*> runtime_record_nodes_;
  std::vector<DeviceRecordNode*> device_record_nodes_;
  void BuildTrees();
  HostRecordNode* BuildTreeRelationship(
      std::vector<HostRecordNode*> host_record_nodes,
      std::vector<CudaRuntimeRecordNode*> runtime_record_nodes);
};

}  // namespace platform
}  // namespace paddle
