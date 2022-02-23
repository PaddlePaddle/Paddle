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

#include "glog/logging.h"

#include "paddle/fluid/platform/profiler/dump/serialization_logger.h"
#include "paddle/fluid/platform/profiler/event_node.h"
#include "paddle/fluid/platform/profiler/extra_info.h"
#include "paddle/fluid/platform/profiler/utils.h"

namespace paddle {
namespace platform {

static const char* kDefaultFilename = "pid_%s_time_%s.paddle_trace.pb";
static const char* version = "1.0.0";
static uint32_t span_indx = 0;

static std::string DefaultFileName() {
  auto pid = GetProcessId();
  return string_format(std::string(kDefaultFilename), pid,
                       GetStringFormatLocalTime().c_str());
}

void SerializationLogger::OpenFile() {
  output_file_stream_.open(filename_, std::ofstream::out |
                                          std::ofstream::trunc |
                                          std::ofstream::binary);
  if (!output_file_stream_) {
    LOG(WARNING) << "Unable to open file for writing profiling data."
                 << std::endl;
  } else {
    LOG(INFO) << "writing profiling data to " << filename_ << std::endl;
  }
  node_trees_proto_ = new NodeTreesProto();
  node_trees_proto_->set_version(std::string(version));
  node_trees_proto_->set_span_indx(span_indx++);
}

void SerializationLogger::LogNodeTrees(const NodeTrees& node_trees) {
  // dump the whole tree into file
  const std::map<uint64_t, std::vector<HostTraceEventNode*>>
      thread2host_event_nodes = node_trees.Traverse(true);

  for (auto it = thread2host_event_nodes.begin();
       it != thread2host_event_nodes.end(); ++it) {
    // 1. order every node an index, every node a parent
    std::map<HostTraceEventNode*, int64_t> node_index_map;
    std::map<HostTraceEventNode*, int64_t> node_parent_map;
    int64_t index = 0;
    for (auto hostnode = it->second.begin(); hostnode != it->second.end();
         ++hostnode) {
      node_index_map[(*hostnode)] = index;  // order each node
      index++;
    }
    node_parent_map[(*(it->second.begin()))] = -1;  // root's parent set as -1
    for (auto hostnode = it->second.begin(); hostnode != it->second.end();
         ++hostnode) {
      for (auto childnode = (*hostnode)->GetChildren().begin();
           childnode != (*hostnode)->GetChildren().end(); ++childnode) {
        node_parent_map[(*childnode)] =
            node_index_map[(*hostnode)];  // mark each node's parent
      }
    }

    // 2. serialize host node, runtime node and device node
    current_thread_node_tree_proto_ =
        node_trees_proto_->add_thread_trees();  // add ThreadNodeTreeProto
    current_thread_node_tree_proto_->set_thread_id(it->first);
    for (auto hostnode = it->second.begin(); hostnode != it->second.end();
         ++hostnode) {
      HostTraceEventNodeProto* host_node_proto =
          current_thread_node_tree_proto_
              ->add_host_nodes();  // add HostTraceEventNodeProto
      host_node_proto->set_id(node_index_map[(*hostnode)]);
      host_node_proto->set_parentid(node_parent_map[(*hostnode)]);
      current_host_trace_event_node_proto_ =
          host_node_proto;       // set current HostTraceEventNodeProto
      (*hostnode)->LogMe(this);  // fill detail information

      for (auto runtimenode = (*hostnode)->GetRuntimeTraceEventNodes().begin();
           runtimenode != (*hostnode)->GetRuntimeTraceEventNodes().end();
           ++runtimenode) {
        CudaRuntimeTraceEventNodeProto* runtime_node_proto =
            current_host_trace_event_node_proto_
                ->add_runtime_nodes();  // add CudaRuntimeTraceEventNodeProto
        current_runtime_trace_event_node_proto_ =
            runtime_node_proto;  // set current CudaRuntimeTraceEventNodeProto
        (*runtimenode)->LogMe(this);  // fill detail information
        for (auto devicenode =
                 (*runtimenode)->GetDeviceTraceEventNodes().begin();
             devicenode != (*runtimenode)->GetDeviceTraceEventNodes().end();
             ++devicenode) {
          DeviceTraceEventNodeProto* device_node_proto =
              current_runtime_trace_event_node_proto_
                  ->add_device_nodes();  // add DeviceTraceEventNodeProto
          current_device_trace_event_node_proto_ =
              device_node_proto;       // set current DeviceTraceEventNodeProto
          (*devicenode)->LogMe(this);  // fill detail information
        }
      }
    }
  }
}

void SerializationLogger::LogHostTraceEventNode(
    const HostTraceEventNode& host_node) {
  HostTraceEventProto* host_trace_event = new HostTraceEventProto();
  host_trace_event->set_name(host_node.Name());
  host_trace_event->set_type(
      static_cast<TracerEventTypeProto>(host_node.Type()));
  host_trace_event->set_start_ns(host_node.StartNs());
  host_trace_event->set_end_ns(host_node.EndNs());
  host_trace_event->set_process_id(host_node.ProcessId());
  host_trace_event->set_thread_id(host_node.ThreadId());
  current_host_trace_event_node_proto_->set_allocated_host_trace_event(
      host_trace_event);
}

void SerializationLogger::LogRuntimeTraceEventNode(
    const CudaRuntimeTraceEventNode& runtime_node) {
  CudaRuntimeTraceEventProto* runtime_trace_event =
      new CudaRuntimeTraceEventProto();
  runtime_trace_event->set_name(runtime_node.Name());
  runtime_trace_event->set_start_ns(runtime_node.StartNs());
  runtime_trace_event->set_end_ns(runtime_node.EndNs());
  runtime_trace_event->set_process_id(runtime_node.ProcessId());
  runtime_trace_event->set_thread_id(runtime_node.ThreadId());
  runtime_trace_event->set_correlation_id(runtime_node.CorrelationId());
  runtime_trace_event->set_callback_id(runtime_node.CallbackId());
  current_runtime_trace_event_node_proto_->set_allocated_runtime_trace_event(
      runtime_trace_event);
}

void SerializationLogger::LogDeviceTraceEventNode(
    const DeviceTraceEventNode& device_node) {
  switch (device_node.Type()) {
    case TracerEventType::Kernel:
      HandleTypeKernel(device_node);
      break;
    case TracerEventType::Memcpy:
      HandleTypeMemcpy(device_node);
      break;
    case TracerEventType::Memset:
      HandleTypeMemset(device_node);
      break;
    default:
      break;
  }
}

void SerializationLogger::HandleTypeKernel(
    const DeviceTraceEventNode& device_node) {
  DeviceTraceEventProto* device_trace_event = new DeviceTraceEventProto();
  KernelEventInfoProto* kernel_info = new KernelEventInfoProto();
  // fill DeviceTraceEventProto
  device_trace_event->set_name(device_node.Name());
  device_trace_event->set_type(
      static_cast<TracerEventTypeProto>(device_node.Type()));
  device_trace_event->set_start_ns(device_node.StartNs());
  device_trace_event->set_end_ns(device_node.EndNs());
  device_trace_event->set_device_id(device_node.DeviceId());
  device_trace_event->set_context_id(device_node.ContextId());
  device_trace_event->set_stream_id(device_node.StreamId());
  device_trace_event->set_correlation_id(device_node.CorrelationId());
  // fill KernelEventInfoProto
  KernelEventInfo info = device_node.KernelInfo();
  kernel_info->set_block_x(info.block_x);
  kernel_info->set_block_y(info.block_y);
  kernel_info->set_block_z(info.block_z);
  kernel_info->set_grid_x(info.grid_x);
  kernel_info->set_grid_y(info.grid_y);
  kernel_info->set_grid_z(info.grid_z);
  kernel_info->set_dynamic_shared_memory(info.dynamic_shared_memory);
  kernel_info->set_static_shared_memory(info.static_shared_memory);
  kernel_info->set_registers_per_thread(info.registers_per_thread);
  kernel_info->set_local_memory_per_thread(info.local_memory_per_thread);
  kernel_info->set_local_memory_total(info.local_memory_total);
  kernel_info->set_queued(info.queued);
  kernel_info->set_submitted(info.submitted);
  kernel_info->set_completed(info.completed);
  // binding
  device_trace_event->set_allocated_kernel_info(kernel_info);
  current_device_trace_event_node_proto_->set_allocated_device_event(
      device_trace_event);
}

void SerializationLogger::HandleTypeMemcpy(
    const DeviceTraceEventNode& device_node) {
  DeviceTraceEventProto* device_trace_event = new DeviceTraceEventProto();
  MemcpyEventInfoProto* memcpy_info = new MemcpyEventInfoProto();
  // fill DeviceTraceEventProto
  device_trace_event->set_name(device_node.Name());
  device_trace_event->set_type(
      static_cast<TracerEventTypeProto>(device_node.Type()));
  device_trace_event->set_start_ns(device_node.StartNs());
  device_trace_event->set_end_ns(device_node.EndNs());
  device_trace_event->set_device_id(device_node.DeviceId());
  device_trace_event->set_context_id(device_node.ContextId());
  device_trace_event->set_stream_id(device_node.StreamId());
  device_trace_event->set_correlation_id(device_node.CorrelationId());
  // fill MemcpyEventInfoProto
  MemcpyEventInfo info = device_node.MemcpyInfo();
  memcpy_info->set_num_bytes(info.num_bytes);
  memcpy_info->set_copy_kind(std::string(info.copy_kind));
  memcpy_info->set_src_kind(std::string(info.src_kind));
  memcpy_info->set_dst_kind(std::string(info.dst_kind));
  // binding
  device_trace_event->set_allocated_memcpy_info(memcpy_info);
  current_device_trace_event_node_proto_->set_allocated_device_event(
      device_trace_event);
}

void SerializationLogger::HandleTypeMemset(
    const DeviceTraceEventNode& device_node) {
  DeviceTraceEventProto* device_trace_event = new DeviceTraceEventProto();
  MemsetEventInfoProto* memset_info = new MemsetEventInfoProto();
  // fill DeviceTraceEventProto
  device_trace_event->set_name(device_node.Name());
  device_trace_event->set_type(
      static_cast<TracerEventTypeProto>(device_node.Type()));
  device_trace_event->set_start_ns(device_node.StartNs());
  device_trace_event->set_end_ns(device_node.EndNs());
  device_trace_event->set_device_id(device_node.DeviceId());
  device_trace_event->set_context_id(device_node.ContextId());
  device_trace_event->set_stream_id(device_node.StreamId());
  device_trace_event->set_correlation_id(device_node.CorrelationId());
  // fill MemsetEventInfoProto
  MemsetEventInfo info = device_node.MemsetInfo();
  memset_info->set_num_bytes(info.num_bytes);
  memset_info->set_memory_kind(std::string(info.memory_kind));
  memset_info->set_value(info.value);
  // binding
  device_trace_event->set_allocated_memset_info(memset_info);
  current_device_trace_event_node_proto_->set_allocated_device_event(
      device_trace_event);
}

void SerializationLogger::LogMetaInfo(
    const std::unordered_map<std::string, std::string> extra_info) {
  for (const auto& kv : extra_info) {
    ExtraInfoMap* extra_info_map = node_trees_proto_->add_extra_info();
    extra_info_map->set_key(kv.first);
    extra_info_map->set_value(kv.second);
  }
}

SerializationLogger::SerializationLogger(const std::string& filename) {
  filename_ = filename.empty() ? DefaultFileName() : filename;
  OpenFile();
}

SerializationLogger::SerializationLogger(const char* filename_cstr) {
  std::string filename(filename_cstr);
  filename_ = filename.empty() ? DefaultFileName() : filename;
  OpenFile();
}

SerializationLogger::~SerializationLogger() {
  if (!output_file_stream_) {
    delete node_trees_proto_;
    return;
  }
  node_trees_proto_->SerializeToOstream(&output_file_stream_);
  delete node_trees_proto_;
  output_file_stream_.close();
}

}  // namespace platform
}  // namespace paddle
