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
#include <map>
#include <string>
#include <vector>

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/profiler/output_logger.h"

namespace paddle {
namespace platform {
class BaseLogger;  // forward declaration

enum class TracerEventType {
  // Used to mark operator record
  Operator = 0,
  // Used to mark dataloader record
  Dataloader = 1,
  // Used to mark profile step record
  ProfileStep = 2,
  // Used to mark cuda runtime record returned by cupti
  CudaRuntime = 3,
  // Used to mark kernel computation record returned by cupti
  Kernel = 4,
  // Used to mark memcpy record returned by cupti
  Memcpy = 5,
  // Used to mark memset record returned by cupti
  Memset = 6,
  // Used to mark record defined by user
  UserDefined = 7,
  // A flag to denote the number of current types
  NumTypes
};

struct KernelRecordInfo {
  // The X-dimension block size for the kernel.
  uint32_t block_x;
  // The Y-dimension block size for the kernel.
  uint32_t block_y;
  // The Z-dimension grid size for the kernel.
  uint32_t block_z;
  // X-dimension of a grid.
  uint32_t grid_x;
  // Y-dimension of a grid.
  uint32_t grid_y;
  // Z-dimension of a grid.
  uint32_t grid_z;
  // The dynamic shared memory reserved for the kernel, in bytes.
  uint32_t dynamic_shared_memory;
  // The static shared memory allocated for the kernel, in bytes.
  uint32_t static_shared_memory;
  // The number of registers required for each thread executing the kernel.
  uint32_t registers_per_thread;
  // The amount of local memory reserved for each thread, in bytes.
  uint32_t local_memory_per_thread;
  // The total amount of local memory reserved for the kernel, in bytes.
  uint32_t local_memory_total;
  // The timestamp when the kernel is queued up in the command buffer, in ns.
  // This timestamp is not collected by default. Use API
  // cuptiActivityEnableLatencyTimestamps() to enable collection.
  uint64_t queued;
  // The timestamp when the command buffer containing the kernel launch is
  // submitted to the GPU, in ns.
  // This timestamp is not collected by default. Use API
  // cuptiActivityEnableLatencyTimestamps() to enable collection.
  uint64_t submitted;
  // The completed timestamp for the kernel execution, in ns.
  uint64_t completed;
};

struct MemcpyRecordInfo {
  // The number of bytes transferred by the memory copy.
  uint64_t num_bytes;
  // The kind of the memory copy.
  // Each kind represents the source and destination targets of a memory copy.
  // Targets are host, device, and array. Refer to CUpti_ActivityMemcpyKind
  uint8_t copy_kind;
  // The source memory kind read by the memory copy.
  // Each kind represents the type of the memory accessed by a memory
  // operation/copy. Refer to CUpti_ActivityMemoryKind
  uint8_t src_kind;
  // The destination memory kind read by the memory copy.
  uint8_t dst_kind;
};

struct MemsetRecordInfo {
  // The number of bytes being set by the memory set.
  uint64_t num_bytes;
  // The memory kind of the memory set. Refer to CUpti_ActivityMemoryKind
  uint8_t memory_kind;
  // the value being assigned to memory by the memory set.
  uint32_t value;
};

class DeviceRecordNode {
 public:
  // constructor
  DeviceRecordNode(const std::string& name, TracerEventType type,
                   uint64_t start_ns, uint64_t end_ns, uint64_t device_id,
                   uint64_t context_id, uint64_t stream_id,
                   uint32_t correlation_id, const KernelRecordInfo& kernel_info)
      : name_(name),
        type_(type),
        start_ns_(start_ns),
        end_ns_(end_ns),
        device_id_(device_id),
        context_id_(context_id),
        stream_id_(stream_id),
        correlation_id_(correlation_id),
        kernel_info_(kernel_info) {}
  DeviceRecordNode(const std::string& name, TracerEventType type,
                   uint64_t start_ns, uint64_t end_ns, uint64_t device_id,
                   uint64_t context_id, uint64_t stream_id,
                   uint32_t correlation_id, const MemcpyRecordInfo& memcpy_info)
      : name_(name),
        type_(type),
        start_ns_(start_ns),
        end_ns_(end_ns),
        device_id_(device_id),
        context_id_(context_id),
        stream_id_(stream_id),
        correlation_id_(correlation_id),
        memcpy_info_(memcpy_info) {}
  DeviceRecordNode(const std::string& name, TracerEventType type,
                   uint64_t start_ns, uint64_t end_ns, uint64_t device_id,
                   uint64_t context_id, uint64_t stream_id,
                   uint32_t correlation_id, const MemsetRecordInfo& memset_info)
      : name_(name),
        type_(type),
        start_ns_(start_ns),
        end_ns_(end_ns),
        device_id_(device_id),
        context_id_(context_id),
        stream_id_(stream_id),
        correlation_id_(correlation_id),
        memset_info_(memset_info) {}

  // destructor
  ~DeviceRecordNode() {}
  // getter
  std::string name() const { return name_; }
  TracerEventType type() const { return type_; }
  uint64_t start_ns() const { return start_ns_; }
  uint64_t end_ns() const { return end_ns_; }
  uint64_t device_id() const { return device_id_; }
  uint64_t context_id() const { return context_id_; }
  uint64_t stream_id() const { return stream_id_; }
  uint64_t duration() const { return end_ns_ - start_ns_; }
  uint32_t correlation_id() const { return correlation_id_; }
  KernelRecordInfo kernel_info() const {
    PADDLE_ENFORCE_EQ(
        type_, TracerEventType::Kernel,
        platform::errors::Unavailable(
            "to get kernel_info, "
            "TracerEventType in node must be TracerEventType::Kernel"));
    return kernel_info_;
  }
  MemcpyRecordInfo memcpy_info() const {
    PADDLE_ENFORCE_EQ(
        type_, TracerEventType::Memcpy,
        platform::errors::Unavailable(
            "to get memcpy_info, "
            "TracerEventType in node must be TracerEventType::Memcpy"));
    return memcpy_info_;
  }
  MemsetRecordInfo memset_info() const {
    PADDLE_ENFORCE_EQ(
        type_, TracerEventType::Memset,
        platform::errors::Unavailable(
            "to get memset_info, "
            "TracerEventType in node must be TracerEventType::Memset"));
    return memset_info_;
  }

  // member function
  void LogMe(BaseLogger* logger) { logger->LogDeviceRecordNode(*this); }

 private:
  // record name
  std::string name_;
  // record type, one of TracerEventType
  TracerEventType type_;
  // start timestamp of the record
  uint64_t start_ns_;
  // end timestamp of the record
  uint64_t end_ns_;
  // device id
  uint64_t device_id_;
  // context id
  uint64_t context_id_;
  // stream id
  uint64_t stream_id_;
  // correlation id, used for correlating async activities happened on device
  uint32_t correlation_id_;
  // union, specific device record type has different detail information
  union {
    // used for TracerEventType::Kernel
    KernelRecordInfo kernel_info_;
    // used for TracerEventType::Memcpy
    MemcpyRecordInfo memcpy_info_;
    // used for TracerEventType::Memset
    MemsetRecordInfo memset_info_;
  };
};

class CudaRuntimeRecordNode {
 public:
  // constructor
  CudaRuntimeRecordNode(const std::string& name, uint64_t start_ns,
                        uint64_t end_ns, uint64_t process_id,
                        uint64_t thread_id, uint32_t correlation_id,
                        uint32_t callback_id)
      : name_(name),
        start_ns_(start_ns),
        end_ns_(end_ns),
        process_id_(process_id),
        thread_id_(thread_id),
        correlation_id_(correlation_id),
        callback_id_(callback_id) {}

  // destructor
  ~CudaRuntimeRecordNode();
  // getter
  std::string name() const { return name_; }
  TracerEventType type() const { return type_; }
  uint64_t start_ns() const { return start_ns_; }
  uint64_t end_ns() const { return end_ns_; }
  uint64_t process_id() const { return process_id_; }
  uint64_t thread_id() const { return thread_id_; }
  uint64_t duration() const { return end_ns_ - start_ns_; }
  uint32_t correlation_id() const { return correlation_id_; }
  uint32_t callback_id() const { return callback_id_; }
  // member function
  void AddDeviceRecordNode(DeviceRecordNode* node) {
    device_node_ptrs_.push_back(node);
  }
  void LogMe(BaseLogger* logger) { logger->LogRuntimeRecordNode(*this); }
  std::vector<DeviceRecordNode*>& GetDeviceRecordNodes() {
    return device_node_ptrs_;
  }

 private:
  // record name
  std::string name_;
  // record type, one of TracerEventType
  TracerEventType type_{TracerEventType::CudaRuntime};
  // start timestamp of the record
  uint64_t start_ns_;
  // end timestamp of the record
  uint64_t end_ns_;
  // process id of the record
  uint64_t process_id_;
  // thread id of the record
  uint64_t thread_id_;
  // correlation id, used for correlating async activities happened on device
  uint32_t correlation_id_;
  // callback id, used to identify which cuda runtime api is called
  uint32_t callback_id_;
  // device records called by this
  std::vector<DeviceRecordNode*> device_node_ptrs_;
};

class HostRecordNode {
 public:
  // constructor
  HostRecordNode(const std::string& name, TracerEventType type,
                 uint64_t start_ns, uint64_t end_ns)
      : name_(name), type_(type), start_ns_(start_ns), end_ns_(end_ns) {}
  HostRecordNode(const std::string& name, TracerEventType type,
                 uint64_t start_ns, uint64_t end_ns, uint64_t process_id,
                 uint64_t thread_id)
      : name_(name),
        type_(type),
        start_ns_(start_ns),
        end_ns_(end_ns),
        process_id_(process_id),
        thread_id_(thread_id) {}

  // destructor
  ~HostRecordNode();

  // getter
  std::string name() const { return name_; }
  TracerEventType type() const { return type_; }
  uint64_t start_ns() const { return start_ns_; }
  uint64_t end_ns() const { return end_ns_; }
  uint64_t process_id() const { return process_id_; }
  uint64_t thread_id() const { return thread_id_; }
  uint64_t duration() const { return end_ns_ - start_ns_; }

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
  // record name
  std::string name_;
  // record type, one of TracerEventType
  TracerEventType type_;
  // start timestamp of the record
  uint64_t start_ns_;
  // end timestamp of the record
  uint64_t end_ns_;
  // process id of the record
  uint64_t process_id_;
  // thread id of the record
  uint64_t thread_id_;
  // cuda runtime records called by this
  std::vector<CudaRuntimeRecordNode*> runtime_node_ptrs_;
  // host records called by this
  std::vector<HostRecordNode*> children_;
};

class NodeTrees {
 public:
  // constructor
  NodeTrees(const std::vector<HostRecordNode*>& host_record_nodes,
            const std::vector<CudaRuntimeRecordNode*>& runtime_record_nodes,
            const std::vector<DeviceRecordNode*>& device_record_nodes) {
    BuildTrees(host_record_nodes, runtime_record_nodes, device_record_nodes);
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
  void BuildTrees(
      const std::vector<HostRecordNode*>& host_record_nodes,
      const std::vector<CudaRuntimeRecordNode*>& runtime_record_nodes,
      const std::vector<DeviceRecordNode*>& device_record_nodes);
  HostRecordNode* BuildTreeRelationship(
      std::vector<HostRecordNode*> host_record_nodes,
      std::vector<CudaRuntimeRecordNode*> runtime_record_nodes);
};

}  // namespace platform
}  // namespace paddle
