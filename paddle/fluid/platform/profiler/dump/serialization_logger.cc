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

#include "paddle/fluid/platform/profiler/dump/serialization_logger.h"

#include "glog/logging.h"
#include "paddle/fluid/platform/profiler/event_node.h"
#include "paddle/fluid/platform/profiler/extra_info.h"
#include "paddle/fluid/platform/profiler/utils.h"

namespace paddle {
namespace platform {

static const char* kDefaultFilename = "pid_%s_time_%s.paddle_trace.pb";

static std::string DefaultFileName() {
  auto pid = GetProcessId();
  return string_format(
      std::string(kDefaultFilename), pid, GetStringFormatLocalTime().c_str());
}

void SerializationLogger::OpenFile() {
  output_file_stream_.open(
      filename_,
      std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
  if (!output_file_stream_) {
    LOG(WARNING) << "Unable to open file for writing profiling data."
                 << std::endl;
  } else {
    LOG(INFO) << "writing profiling data to " << filename_ << std::endl;
  }
  node_trees_proto_ = new NodeTreesProto();
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
void SerializationLogger::LogDeviceProperty(
    const std::map<uint32_t, gpuDeviceProp>& device_property_map) {
  for (auto it = device_property_map.begin(); it != device_property_map.end();
       it++) {
    const gpuDeviceProp& device_property = it->second;
    DevicePropertyProto* device_property_proto =
        node_trees_proto_->add_device_property();
    device_property_proto->set_id(it->first);
    device_property_proto->set_name(device_property.name);
    device_property_proto->set_total_global_memory(
        device_property.totalGlobalMem);
    device_property_proto->set_compute_major(device_property.major);
    device_property_proto->set_compute_minor(device_property.minor);
    device_property_proto->set_sm_count(device_property.multiProcessorCount);
#if defined(PADDLE_WITH_CUDA)
    device_property_proto->set_max_threads_per_block(
        device_property.maxThreadsPerBlock);
    device_property_proto->set_max_threads_per_multiprocessor(
        device_property.maxThreadsPerMultiProcessor);
    device_property_proto->set_regs_per_block(device_property.regsPerBlock);
    device_property_proto->set_regs_per_multiprocessor(
        device_property.regsPerMultiprocessor);
    device_property_proto->set_warp_size(device_property.warpSize);
    device_property_proto->set_shared_memory_per_block(
        device_property.sharedMemPerBlock);
    device_property_proto->set_shared_memory_per_multiprocessor(
        device_property.sharedMemPerMultiprocessor);
    device_property_proto->set_shared_memory_per_block_optin(
        device_property.sharedMemPerBlockOptin);
#endif
  }
}
#endif

void SerializationLogger::LogNodeTrees(const NodeTrees& node_trees) {
  // dump the whole tree into file
  const std::map<uint64_t, std::vector<HostTraceEventNode*>>
      thread2host_event_nodes = node_trees.Traverse(true);

  for (auto it = thread2host_event_nodes.begin();
       it != thread2host_event_nodes.end();
       ++it) {
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
           childnode != (*hostnode)->GetChildren().end();
           ++childnode) {
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
      for (auto memnode = (*hostnode)->GetMemTraceEventNodes().begin();
           memnode != (*hostnode)->GetMemTraceEventNodes().end();
           ++memnode) {
        MemTraceEventNodeProto* mem_node_proto =
            current_host_trace_event_node_proto_->add_mem_nodes();
        current_mem_trace_event_node_proto_ = mem_node_proto;
        (*memnode)->LogMe(this);
      }
    }
  }
}

void SerializationLogger::LogMemTraceEventNode(
    const MemTraceEventNode& mem_node) {
  MemTraceEventProto* mem_trace_event = new MemTraceEventProto();
  mem_trace_event->set_timestamp_ns(mem_node.TimeStampNs());
  mem_trace_event->set_type(
      static_cast<TracerMemEventTypeProto>(mem_node.Type()));
  mem_trace_event->set_addr(mem_node.Addr());
  mem_trace_event->set_process_id(mem_node.ProcessId());
  mem_trace_event->set_thread_id(mem_node.ThreadId());
  mem_trace_event->set_increase_bytes(mem_node.IncreaseBytes());
  mem_trace_event->set_place(mem_node.Place());
  mem_trace_event->set_current_allocated(mem_node.CurrentAllocated());
  mem_trace_event->set_current_reserved(mem_node.CurrentReserved());
  mem_trace_event->set_peak_allocated(mem_node.PeakAllocated());
  mem_trace_event->set_peak_reserved(mem_node.PeakReserved());
  current_mem_trace_event_node_proto_->set_allocated_mem_event(mem_trace_event);
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
  OperatorSupplementEventNode* op_supplement_event_node =
      host_node.GetOperatorSupplementEventNode();
  if (op_supplement_event_node != nullptr) {
    current_op_supplement_event_node_proto_ =
        current_host_trace_event_node_proto_->add_op_supplement_nodes();
    OperatorSupplementEventProto* op_supplement_event_proto =
        new OperatorSupplementEventProto();
    op_supplement_event_proto->set_op_type(op_supplement_event_node->Name());
    op_supplement_event_proto->set_timestamp_ns(
        op_supplement_event_node->TimeStampNs());
    op_supplement_event_proto->set_process_id(
        op_supplement_event_node->ProcessId());
    op_supplement_event_proto->set_thread_id(
        op_supplement_event_node->ThreadId());
    op_supplement_event_proto->set_callstack(
        op_supplement_event_node->CallStack());

    OperatorSupplementEventProto::input_shape_proto* input_shape_proto =
        op_supplement_event_proto->mutable_input_shapes();
    for (auto it = op_supplement_event_node->InputShapes().begin();
         it != op_supplement_event_node->InputShapes().end();
         it++) {
      input_shape_proto->add_key(it->first);
      OperatorSupplementEventProto::input_shape_proto::shape_vector*
          shape_vectors_proto = input_shape_proto->add_shape_vecs();
      auto shape_vectors = it->second;
      for (auto shape_vecs_it = shape_vectors.begin();
           shape_vecs_it != shape_vectors.end();
           shape_vecs_it++) {
        auto shape_vector = *shape_vecs_it;
        OperatorSupplementEventProto::input_shape_proto::shape_vector::shape*
            shape_proto = shape_vectors_proto->add_shapes();
        for (auto shape_it = shape_vector.begin();
             shape_it != shape_vector.end();
             shape_it++) {
          shape_proto->add_size(*shape_it);
        }
      }
    }

    OperatorSupplementEventProto::dtype_proto* dtype_proto =
        op_supplement_event_proto->mutable_dtypes();
    for (auto it = op_supplement_event_node->Dtypes().begin();
         it != op_supplement_event_node->Dtypes().end();
         it++) {
      dtype_proto->add_key(it->first);
      OperatorSupplementEventProto::dtype_proto::dtype_vector*
          dtype_vector_proto = dtype_proto->add_dtype_vecs();
      auto dtype_vector = it->second;
      for (auto dtype_it = dtype_vector.begin(); dtype_it != dtype_vector.end();
           dtype_it++) {
        dtype_vector_proto->add_dtype(*dtype_it);
      }
    }
    current_op_supplement_event_node_proto_->set_allocated_op_supplement_event(
        op_supplement_event_proto);
  }
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
  kernel_info->set_blocks_per_sm(info.blocks_per_sm);
  kernel_info->set_warps_per_sm(info.warps_per_sm);
  kernel_info->set_occupancy(info.occupancy);
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

void SerializationLogger::LogExtraInfo(
    const std::unordered_map<std::string, std::string> extra_info) {
  for (const auto& kv : extra_info) {
    ExtraInfoMap* extra_info_map = node_trees_proto_->add_extra_info();
    extra_info_map->set_key(kv.first);
    extra_info_map->set_value(kv.second);
  }
}

void SerializationLogger::LogMetaInfo(const std::string& version,
                                      uint32_t span_indx) {
  node_trees_proto_->set_version(version);
  node_trees_proto_->set_span_indx(span_indx);
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
