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

#include <string>
#include <vector>
#include <map>
#include <functional>

#include "paddle/fluid/platform/enforce.h"

namespace paddle{
namespace platform {

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

struct KernelDetails {
  // the number of registers used in this kernel.
  uint32_t registers_per_thread;
  // the amount of shared memory space used by a thread block.
  uint32_t static_shared_memory_usage;
  // the amount of dynamic memory space used by a thread block.
  uint32_t dynamic_shared_memory_usage;
  // X-dimension of a thread block.
  uint32_t block_x;
  // Y-dimension of a thread block.
  uint32_t block_y;
  // Z-dimension of a thread block.
  uint32_t block_z;
  // X-dimension of a grid.
  uint32_t grid_x;
  // Y-dimension of a grid.
  uint32_t grid_y;
  // Z-dimension of a grid.
  uint32_t grid_z;
};

struct MemcpyDetails {
  // the amount of data copied for memcpy events.
  size_t num_bytes;
  // the destination device id for peer-2-peer communication (memcpy). The source
  // device is implicit: it's the current device.
  uint32_t destination_device_id;
  // whether or not the memcpy is asynchronous.
  bool async;
  // this contains CUpti_ActivityMemcpyKind for activity event (on device).
  // for events from other CuptiTracerEventSource, it is always 0.
  int8_t copy_kind;
  // CUpti_ActivityMemoryKind of source.
  int8_t src_mem_kind;
  // CUpti_ActivityMemoryKind of destination.
  int8_t dst_mem_kind;
};

struct MemsetDetails {
  // size of memory to be written over in bytes.
  size_t num_bytes;
  // the CUpti_ActivityMemoryKind value for this activity event.
  int8_t mem_kind;
  // whether or not the memset is asynchronous.
  bool async;
  // the value being assigned to memory by the memory set.
  uint32_t value;

};

class HostRecordNode {
  public:
    // constructor
    HostRecordNode(const std::string &name, TracerEventType type,
                  uint64_t start_ns, uint64_t end_ns):
                  name_(name), type_(type), start_ns_(start_ns),
                  end_ns_(end_ns) {}
    HostRecordNode(const std::string &name, TracerEventType type,
                   uint64_t start_ns, uint64_t end_ns,
                   uint64_t process_id, uint64_t thread_id):
                   name_(name), type_(type), start_ns_(start_ns),
                   end_ns_(end_ns), process_id_(process_id), thread_id_(thread_id){}
    
    // destructor
    ~HostRecordNode();

    // getter
    std::string name() { return name_; }
    TracerEventType type() { return type_; }
    uint64_t start_ns() { return start_ns_; }
    uint64_t end_ns() { return end_ns_; }
    uint64_t process_id() { return process_id_; }
    uint64_t thread_id_() { return thread_id_; }
    uint64_t duration() { return end_ns_ - start_ns_; }

    // member function
    void AddChild(HostRecordNode* node);
    void AddCudaRuntimeNode(CudaRuntimeRecordNode* node);
    std::vector<HostRecordNode*> GetChildren();
    std::vector<CudaRuntimeRecordNode*> GetRuntimeRecordNodes();
    void logme(BaseLogger* logger);

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

class CudaRuntimeRecordNode{
  public:
    // constructor
    CudaRuntimeRecordNode(const std::string &name, 
                          uint64_t start_ns, uint64_t end_ns,
                          uint64_t process_id, uint64_t thread_id,
                          uint32_t correlation_id, uint32_t callback_id):
                          name_(name), type_(type), start_ns_(start_ns),
                          end_ns_(end_ns), process_id_(process_id), 
                          thread_id_(thread_id), correlation_id_(correlation_id),
                          callback_id_(callback_id) {}
    
    // destructor
    ~CudaRuntimeRecordNode();
    // getter
    std::string name() { return name_; }
    TracerEventType type() { return type_; }
    uint64_t start_ns() { return start_ns_; }
    uint64_t end_ns() { return end_ns_; }
    uint64_t process_id() { return process_id_; }
    uint64_t thread_id_() { return thread_id_; }
    uint64_t duration() { return end_ns_ - start_ns_; }
    uint32_t correlation_id() { return correlation_id_; }
    uint32_t callback_id() {return callback_id_; }
    // member function
    void AddDeviceRecordNode(DeviceRecordNode* node);
    void logme(BaseLogger* logger);
    std::vector<DeviceRecordNode*> GetDeviceRecordNodes();

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

class DeviceRecordNode{
  public:
    // constructor
    DeviceRecordNode(const std::string &name, TracerEventType type,
                     uint64_t start_ns, uint64_t end_ns,
                     uint64_t device_id, uint64_t context_id,
                     uint64_t stream_id, uint32_t correlation_id,
                     const KernelDetails& kernel_info):
                     name_(name), type_(type), start_ns_(start_ns),
                     end_ns_(end_ns), device_id_(device_id), 
                     context_id_(context_id), stream_id_(stream_id)
                     correlation_id_(correlation_id), kernel_info_(kernel_info) {}
    DeviceRecordNode(const std::string &name, TracerEventType type,
                     uint64_t start_ns, uint64_t end_ns,
                     uint64_t device_id, uint64_t context_id,
                     uint64_t stream_id, uint32_t correlation_id,
                     const MemcpyDetails& memcpy_info):
                     name_(name), type_(type), start_ns_(start_ns),
                     end_ns_(end_ns), device_id_(device_id), 
                     context_id_(context_id), stream_id_(stream_id)
                     correlation_id_(correlation_id), memcpy_info_(memcpy_info) {}
    DeviceRecordNode(const std::string &name, TracerEventType type,
                     uint64_t start_ns, uint64_t end_ns,
                     uint64_t device_id, uint64_t context_id,
                     uint64_t stream_id, uint32_t correlation_id,
                     const MemsetDetails& memset_info):
                     name_(name), type_(type), start_ns_(start_ns),
                     end_ns_(end_ns), device_id_(device_id), 
                     context_id_(context_id), stream_id_(stream_id)
                     correlation_id_(correlation_id), memset_info_(memset_info) {}
    
    // destructor
    ~DeviceRecordNode();
    // getter
    std::string name() { return name_; }
    TracerEventType type() { return type_; }
    uint64_t start_ns() { return start_ns_; }
    uint64_t end_ns() { return end_ns_; }
    uint64_t device_id() { return device_id_; }
    uint64_t context_id() { return context_id_; }
    uint64_t stream_id() { return stream_id_; }
    uint64_t duration() { return end_ns_ - start_ns_; }
    uint32_t correlation_id() { return correlation_id_; }
    KernelDetails kernel_info() { 
      PADDLE_ENFORCE_EQ(type_, TracerEventType::Kernel, 
                        platform::errors::Unavailable("to get kernel_info, \
                        TracerEventType in node must be TracerEventType::Kernel")
                      );
      return kernel_info_; 
      }
    MemcpyDetails memcpy_info() { 
      PADDLE_ENFORCE_EQ(type_, TracerEventType::Memcpy, 
                        platform::errors::Unavailable("to get memcpy_info, \
                        TracerEventType in node must be TracerEventType::Memcpy")
                      );
      return memcpy_info_; 
      }
    MemsetDetails memset_info() { 
      PADDLE_ENFORCE_EQ(type_, TracerEventType::Memset, 
                        platform::errors::Unavailable( "to get memset_info, \
                        TracerEventType in node must be TracerEventType::Memset")
                      );
      return memset_info_; 
      }
    
    // member function
    void logme(BaseLogger* logger);

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
      KernelDetails kernel_info_;
      // used for TracerEventType::Memcpy
      MemcpyDetails memcpy_info_;
      // used for TracerEventType::Memset
      MemsetDetails memset_info_;
    }   
};


class NodeTrees {
  public:
    // constructor
    NodeTrees(const std::vector<HostRecordNode*>& host_record_nodes, 
              const std::vector<CudaRuntimeRecordNode*>& runtime_record_nodes,
              const std::vector<DeviceRecordNode*>& device_record_nodes) 
              {
                BuildTrees(host_record_nodes, runtime_record_nodes, device_record_nodes);
              };
    
    // destructor
    ~NodeTrees();
    
    void logme(BaseLogger* logger);
    void traverse(std::function<void (HostRecordNode*)>, 
                  std::function<void (CudaRuntimeRecordNode*)>, 
                  std::function<void (DeviceRecordNode*)>);
    const std::map<uint64_t, HostRecordNode*>& GetNodeTrees() { return thread_record_trees_map_; }
    

  private:
    std::map<uint64_t, HostRecordNode*> thread_record_trees_map_;
    bool BuildTrees(const std::vector<HostRecordNode*>& host_record_nodes, 
              const std::vector<CudaRuntimeRecordNode*>& runtime_record_nodes,
              const std::vector<DeviceRecordNode*>& device_record_nodes);

};

} // namespace platform
} // namespace paddle

