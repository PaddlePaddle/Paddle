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
#include "paddle/fluid/platform/profiler/dump/deserialization_reader.h"
#include <cstring>
#include "paddle/fluid/platform/profiler/extra_info.h"

namespace paddle {
namespace platform {

DeserializationReader::DeserializationReader(const std::string& filename)
    : filename_(filename) {
  OpenFile();
  node_trees_proto_ = new NodeTreesProto();
}

DeserializationReader::DeserializationReader(const char* filename)
    : filename_(filename) {
  OpenFile();
  node_trees_proto_ = new NodeTreesProto();
}

void DeserializationReader::OpenFile() {
  input_file_stream_.open(filename_, std::ifstream::in | std::ifstream::binary);
  if (!input_file_stream_) {
    VLOG(2) << "Unable to open file for writing profiling data." << std::endl;
  } else {
    VLOG(0) << "Read profiling data from " << filename_ << std::endl;
  }
}

std::unique_ptr<ProfilerResult> DeserializationReader::Parse() {
  if (!node_trees_proto_->ParseFromIstream(&input_file_stream_)) {
    VLOG(2) << "Unable to load node trees in protobuf." << std::endl;
    return nullptr;
  }
  // restore extra info
  ExtraInfo extrainfo;
  for (auto indx = 0; indx < node_trees_proto_->extra_info_size(); indx++) {
    ExtraInfoMap extra_info_map = node_trees_proto_->extra_info(indx);
    extrainfo.AddExtraInfo(extra_info_map.key(), std::string("%s"),
                           extra_info_map.value().c_str());
  }
  // restore NodeTrees
  std::map<uint64_t, HostTraceEventNode*> thread_event_trees_map;
  for (int node_tree_index = 0;
       node_tree_index < node_trees_proto_->thread_trees_size();
       node_tree_index++) {
    // handle one thread tree
    std::map<int64_t, HostTraceEventNode*> index_node_map;
    std::map<int64_t, int64_t> child_parent_map;
    const ThreadNodeTreeProto& thread_node_tree_proto =
        node_trees_proto_->thread_trees(node_tree_index);
    uint64_t current_threadid = thread_node_tree_proto.thread_id();
    for (int host_node_index = 0;
         host_node_index < thread_node_tree_proto.host_nodes_size();
         host_node_index++) {
      // handle host node
      const HostTraceEventNodeProto& host_node_proto =
          thread_node_tree_proto.host_nodes(host_node_index);
      HostTraceEventNode* host_node =
          RestoreHostTraceEventNode(host_node_proto);
      index_node_map[host_node_proto.id()] = host_node;
      child_parent_map[host_node_proto.id()] = host_node_proto.parentid();
      // handle runtime node
      for (int runtime_node_index = 0;
           runtime_node_index < host_node_proto.runtime_nodes_size();
           runtime_node_index++) {
        const CudaRuntimeTraceEventNodeProto& runtime_node_proto =
            host_node_proto.runtime_nodes(runtime_node_index);
        CudaRuntimeTraceEventNode* runtime_node =
            RestoreCudaRuntimeTraceEventNode(runtime_node_proto);
        host_node->AddCudaRuntimeNode(runtime_node);  // insert into host_node
        // handle device node
        for (int device_node_index = 0;
             device_node_index < runtime_node_proto.device_nodes_size();
             device_node_index++) {
          const DeviceTraceEventNodeProto& device_node_proto =
              runtime_node_proto.device_nodes(device_node_index);
          DeviceTraceEventNode* device_node =
              RestoreDeviceTraceEventNode(device_node_proto);
          runtime_node->AddDeviceTraceEventNode(
              device_node);  // insert into runtime_node
        }
      }
    }
    // restore parent-child relationship
    for (auto it = child_parent_map.begin(); it != child_parent_map.end();
         it++) {
      if (it->second != -1) {  // not root node
        index_node_map[it->second]->AddChild(index_node_map[it->first]);
      } else {
        thread_event_trees_map[current_threadid] =
            index_node_map[it->first];  // root node
      }
    }
  }
  // restore NodeTrees object
  std::unique_ptr<NodeTrees> tree(new NodeTrees(thread_event_trees_map));
  return std::unique_ptr<ProfilerResult>(
      new ProfilerResult(std::move(tree), extrainfo));
}

DeserializationReader::~DeserializationReader() {
  delete node_trees_proto_;
  input_file_stream_.close();
}

DeviceTraceEventNode* DeserializationReader::RestoreDeviceTraceEventNode(
    const DeviceTraceEventNodeProto& device_node_proto) {
  const DeviceTraceEventProto& device_event_proto =
      device_node_proto.device_event();
  DeviceTraceEvent device_event;
  device_event.name = device_event_proto.name();
  device_event.type = static_cast<TracerEventType>(device_event_proto.type());
  device_event.start_ns = device_event_proto.start_ns();
  device_event.end_ns = device_event_proto.end_ns();
  device_event.device_id = device_event_proto.device_id();
  device_event.context_id = device_event_proto.context_id();
  device_event.stream_id = device_event_proto.stream_id();
  device_event.correlation_id = device_event_proto.correlation_id();
  switch (device_event.type) {
    case TracerEventType::Kernel:
      device_event.kernel_info = HandleKernelEventInfoProto(device_event_proto);
      break;

    case TracerEventType::Memcpy:
      device_event.memcpy_info = HandleMemcpyEventInfoProto(device_event_proto);
      break;

    case TracerEventType::Memset:
      device_event.memset_info = HandleMemsetEventInfoProto(device_event_proto);
      break;
    default:
      break;
  }
  return new DeviceTraceEventNode(device_event);
}

CudaRuntimeTraceEventNode*
DeserializationReader::RestoreCudaRuntimeTraceEventNode(
    const CudaRuntimeTraceEventNodeProto& runtime_node_proto) {
  const CudaRuntimeTraceEventProto& runtime_event_proto =
      runtime_node_proto.runtime_trace_event();
  RuntimeTraceEvent runtime_event;
  runtime_event.name = runtime_event_proto.name();
  runtime_event.start_ns = runtime_event_proto.start_ns();
  runtime_event.end_ns = runtime_event_proto.end_ns();
  runtime_event.process_id = runtime_event_proto.process_id();
  runtime_event.thread_id = runtime_event_proto.thread_id();
  runtime_event.correlation_id = runtime_event_proto.correlation_id();
  runtime_event.callback_id = runtime_event_proto.callback_id();
  return new CudaRuntimeTraceEventNode(runtime_event);
}

HostTraceEventNode* DeserializationReader::RestoreHostTraceEventNode(
    const HostTraceEventNodeProto& host_node_proto) {
  const HostTraceEventProto& host_event_proto =
      host_node_proto.host_trace_event();
  HostTraceEvent host_event;
  host_event.name = host_event_proto.name();
  host_event.type = static_cast<TracerEventType>(host_event_proto.type());
  host_event.start_ns = host_event_proto.start_ns();
  host_event.end_ns = host_event_proto.end_ns();
  host_event.process_id = host_event_proto.process_id();
  host_event.thread_id = host_event_proto.thread_id();
  return new HostTraceEventNode(host_event);
}

KernelEventInfo DeserializationReader::HandleKernelEventInfoProto(
    const DeviceTraceEventProto& device_event_proto) {
  const KernelEventInfoProto& kernel_info_proto =
      device_event_proto.kernel_info();
  KernelEventInfo kernel_info;
  kernel_info.block_x = kernel_info_proto.block_x();
  kernel_info.block_y = kernel_info_proto.block_y();
  kernel_info.block_z = kernel_info_proto.block_z();
  kernel_info.grid_x = kernel_info_proto.grid_x();
  kernel_info.grid_y = kernel_info_proto.grid_y();
  kernel_info.grid_z = kernel_info_proto.grid_z();
  kernel_info.dynamic_shared_memory = kernel_info_proto.dynamic_shared_memory();
  kernel_info.static_shared_memory = kernel_info_proto.static_shared_memory();
  kernel_info.registers_per_thread = kernel_info_proto.registers_per_thread();
  kernel_info.local_memory_per_thread =
      kernel_info_proto.local_memory_per_thread();
  kernel_info.local_memory_total = kernel_info_proto.local_memory_total();
  kernel_info.queued = kernel_info_proto.queued();
  kernel_info.submitted = kernel_info_proto.submitted();
  kernel_info.completed = kernel_info_proto.completed();
  return kernel_info;
}

MemcpyEventInfo DeserializationReader::HandleMemcpyEventInfoProto(
    const DeviceTraceEventProto& device_event_proto) {
  const MemcpyEventInfoProto& memcpy_info_proto =
      device_event_proto.memcpy_info();
  MemcpyEventInfo memcpy_info;
  memcpy_info.num_bytes = memcpy_info_proto.num_bytes();
  std::strncpy(memcpy_info.copy_kind, memcpy_info_proto.copy_kind().c_str(),
               kMemKindMaxLen - 1);
  std::strncpy(memcpy_info.src_kind, memcpy_info_proto.src_kind().c_str(),
               kMemKindMaxLen - 1);
  std::strncpy(memcpy_info.dst_kind, memcpy_info_proto.dst_kind().c_str(),
               kMemKindMaxLen - 1);
  return memcpy_info;
}

MemsetEventInfo DeserializationReader::HandleMemsetEventInfoProto(
    const DeviceTraceEventProto& device_event_proto) {
  const MemsetEventInfoProto& memset_info_proto =
      device_event_proto.memset_info();
  MemsetEventInfo memset_info;
  memset_info.num_bytes = memset_info_proto.num_bytes();
  std::strncpy(memset_info.memory_kind, memset_info_proto.memory_kind().c_str(),
               kMemKindMaxLen - 1);
  memset_info.value = memset_info_proto.value();
  return memset_info;
}

}  // namespace platform
}  // namespace paddle
