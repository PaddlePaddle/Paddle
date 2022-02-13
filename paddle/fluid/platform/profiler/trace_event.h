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

#include <string>

namespace paddle {
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

struct KernelEventInfo {
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

static constexpr size_t kMemKindMaxLen = 50;

struct MemcpyEventInfo {
  // The number of bytes transferred by the memory copy.
  uint64_t num_bytes;
  // The kind of the memory copy.
  // Each kind represents the source and destination targets of a memory copy.
  // Targets are host, device, and array. Refer to CUpti_ActivityMemcpyKind
  char copy_kind[kMemKindMaxLen];
  // The source memory kind read by the memory copy.
  // Each kind represents the type of the memory accessed by a memory
  // operation/copy. Refer to CUpti_ActivityMemoryKind
  char src_kind[kMemKindMaxLen];
  // The destination memory kind read by the memory copy.
  char dst_kind[kMemKindMaxLen];
};

struct MemsetEventInfo {
  // The number of bytes being set by the memory set.
  uint64_t num_bytes;
  // The memory kind of the memory set. Refer to CUpti_ActivityMemoryKind
  char memory_kind[kMemKindMaxLen];
  // the value being assigned to memory by the memory set.
  uint32_t value;
};

struct HostTraceEvent {
  HostTraceEvent() = default;
  HostTraceEvent(const std::string& name, TracerEventType type,
                 uint64_t start_ns, uint64_t end_ns, uint64_t process_id,
                 uint64_t thread_id)
      : name(name),
        type(type),
        start_ns(start_ns),
        end_ns(end_ns),
        process_id(process_id),
        thread_id(thread_id) {}
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
};

struct RuntimeTraceEvent {
  RuntimeTraceEvent() = default;
  RuntimeTraceEvent(const std::string& name, uint64_t start_ns, uint64_t end_ns,
                    uint64_t process_id, uint64_t thread_id,
                    uint32_t correlation_id, uint32_t callback_id)
      : name(name),
        start_ns(start_ns),
        end_ns(end_ns),
        process_id(process_id),
        thread_id(thread_id),
        correlation_id(correlation_id),
        callback_id(callback_id) {}

  // record name
  std::string name;
  // record type, one of TracerEventType
  TracerEventType type{TracerEventType::CudaRuntime};
  // start timestamp of the record
  uint64_t start_ns;
  // end timestamp of the record
  uint64_t end_ns;
  // process id of the record
  uint64_t process_id;
  // thread id of the record
  uint64_t thread_id;
  // correlation id, used for correlating async activities happened on device
  uint32_t correlation_id;
  // callback id, used to identify which cuda runtime api is called
  uint32_t callback_id;
};

struct DeviceTraceEvent {
  DeviceTraceEvent() = default;
  DeviceTraceEvent(const std::string& name, TracerEventType type,
                   uint64_t start_ns, uint64_t end_ns, uint64_t device_id,
                   uint64_t context_id, uint64_t stream_id,
                   uint32_t correlation_id, const KernelEventInfo& kernel_info)
      : name(name),
        type(type),
        start_ns(start_ns),
        end_ns(end_ns),
        device_id(device_id),
        context_id(context_id),
        stream_id(stream_id),
        correlation_id(correlation_id),
        kernel_info(kernel_info) {}
  DeviceTraceEvent(const std::string& name, TracerEventType type,
                   uint64_t start_ns, uint64_t end_ns, uint64_t device_id,
                   uint64_t context_id, uint64_t stream_id,
                   uint32_t correlation_id, const MemcpyEventInfo& memcpy_info)
      : name(name),
        type(type),
        start_ns(start_ns),
        end_ns(end_ns),
        device_id(device_id),
        context_id(context_id),
        stream_id(stream_id),
        correlation_id(correlation_id),
        memcpy_info(memcpy_info) {}
  DeviceTraceEvent(const std::string& name, TracerEventType type,
                   uint64_t start_ns, uint64_t end_ns, uint64_t device_id,
                   uint64_t context_id, uint64_t stream_id,
                   uint32_t correlation_id, const MemsetEventInfo& memset_info)
      : name(name),
        type(type),
        start_ns(start_ns),
        end_ns(end_ns),
        device_id(device_id),
        context_id(context_id),
        stream_id(stream_id),
        correlation_id(correlation_id),
        memset_info(memset_info) {}
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
  // correlation id, used for correlating async activities happened on device
  uint32_t correlation_id;
  // union, specific device record type has different detail information
  union {
    // used for TracerEventType::Kernel
    KernelEventInfo kernel_info;
    // used for TracerEventType::Memcpy
    MemcpyEventInfo memcpy_info;
    // used for TracerEventType::Memset
    MemsetEventInfo memset_info;
  };
};

}  // namespace platform
}  // namespace paddle
