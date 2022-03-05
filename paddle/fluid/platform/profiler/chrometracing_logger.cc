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

#include <cstdio>
#include <ctime>

#include "glog/logging.h"

#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/profiler/chrometracing_logger.h"
#include "paddle/fluid/platform/profiler/event_node.h"
#include "paddle/fluid/platform/profiler/utils.h"

namespace paddle {
namespace platform {

static const char* kSchemaVersion = "1.0.0";
static const char* kDefaultFilename = "pid_%s_time_%s.paddle_trace.json";
static uint32_t span_indx = 0;

static std::string DefaultFileName() {
  auto pid = GetProcessId();
  return string_format(std::string(kDefaultFilename), pid,
                       GetStringFormatLocalTime().c_str());
}

const char* ChromeTracingLogger::categary_name_[] = {
    "Operator",      "Dataloader", "ProfileStep",      "CudaRuntime",
    "Kernel",        "Memcpy",     "Memset",           "UserDefined",
    "OperatorInner", "Forward",    "Backward",         "Optimization",
    "Communication", "PythonOp",   "PythonUserDefined"};

void ChromeTracingLogger::OpenFile() {
  output_file_stream_.open(filename_,
                           std::ofstream::out | std::ofstream::trunc);
  if (!output_file_stream_) {
    LOG(WARNING) << "Unable to open file for writing profiling data."
                 << std::endl;
  } else {
    LOG(INFO) << "writing profiling data to " << filename_ << std::endl;
  }
}

ChromeTracingLogger::ChromeTracingLogger(const std::string& filename) {
  filename_ = filename.empty() ? DefaultFileName() : filename;
  OpenFile();
  StartLog();
}

ChromeTracingLogger::ChromeTracingLogger(const char* filename_cstr) {
  std::string filename(filename_cstr);
  filename_ = filename.empty() ? DefaultFileName() : filename;
  OpenFile();
  StartLog();
}

ChromeTracingLogger::~ChromeTracingLogger() {
  EndLog();
  output_file_stream_.close();
}

void ChromeTracingLogger::LogNodeTrees(const NodeTrees& node_trees) {
  // log all nodes except root node, root node is a helper node.
  const std::map<uint64_t, std::vector<HostTraceEventNode*>>
      thread2host_event_nodes = node_trees.Traverse(true);
  for (auto it = thread2host_event_nodes.begin();
       it != thread2host_event_nodes.end(); ++it) {
    for (auto hostnode = it->second.begin(); hostnode != it->second.end();
         ++hostnode) {
      if (hostnode != it->second.begin()) {  // skip root node
        (*hostnode)->LogMe(this);
      }
      for (auto runtimenode = (*hostnode)->GetRuntimeTraceEventNodes().begin();
           runtimenode != (*hostnode)->GetRuntimeTraceEventNodes().end();
           ++runtimenode) {
        (*runtimenode)->LogMe(this);
        for (auto devicenode =
                 (*runtimenode)->GetDeviceTraceEventNodes().begin();
             devicenode != (*runtimenode)->GetDeviceTraceEventNodes().end();
             ++devicenode) {
          (*devicenode)->LogMe(this);
        }
      }
    }
  }
}

void ChromeTracingLogger::LogHostTraceEventNode(
    const HostTraceEventNode& host_node) {
  if (!output_file_stream_) {
    return;
  }
  switch (host_node.Type()) {
    case TracerEventType::ProfileStep:
    case TracerEventType::Forward:
    case TracerEventType::Backward:
    case TracerEventType::Dataloader:
    case TracerEventType::Optimization:
    case TracerEventType::PythonOp:
    case TracerEventType::PythonUserDefined:
      output_file_stream_ << string_format(
          std::string(
              R"JSON(
  { 
    "name": "%s", "pid": %lld, "tid": "%lld(Python)",
    "ts": %lld, "dur": %lld,
    "ph": "X", "cat": "%s", 
    "args": {
      "start_ns": %lld,
      "end_ns": %lld
    }
  },
  )JSON"),
          host_node.Name().c_str(), host_node.ProcessId(), host_node.ThreadId(),
          nsToUs(host_node.StartNs()), nsToUs(host_node.Duration()),
          categary_name_[static_cast<int>(host_node.Type())],
          host_node.StartNs(), host_node.EndNs());
      break;
    default:
      output_file_stream_ << string_format(
          std::string(
              R"JSON(
  { 
    "name": "%s", "pid": %lld, "tid": "%lld(C++)",
    "ts": %lld, "dur": %lld,
    "ph": "X", "cat": "%s", 
    "args": {
      "start_ns": %lld,
      "end_ns": %lld
    }
  },
  )JSON"),
          host_node.Name().c_str(), host_node.ProcessId(), host_node.ThreadId(),
          nsToUs(host_node.StartNs()), nsToUs(host_node.Duration()),
          categary_name_[static_cast<int>(host_node.Type())],
          host_node.StartNs(), host_node.EndNs());
      break;
  }

  pid_tid_set_.insert({host_node.ProcessId(), host_node.ThreadId()});
}

void ChromeTracingLogger::LogRuntimeTraceEventNode(
    const CudaRuntimeTraceEventNode& runtime_node) {
  if (!output_file_stream_) {
    return;
  }
  output_file_stream_ << string_format(
      std::string(
          R"JSON(
  { 
    "name": "%s", "pid": %lld, "tid": "%lld(C++)",
    "ts": %lld, "dur": %lld,
    "ph": "X", "cat": "%s", 
    "args": {
      "correlation id": %d,
      "start_ns": %lld,
      "end_ns": %lld
    }
  },
  )JSON"),
      runtime_node.Name().c_str(), runtime_node.ProcessId(),
      runtime_node.ThreadId(), nsToUs(runtime_node.StartNs()),
      nsToUs(runtime_node.Duration()),
      categary_name_[static_cast<int>(runtime_node.Type())],
      runtime_node.CorrelationId(), runtime_node.StartNs(),
      runtime_node.EndNs());
  pid_tid_set_.insert({runtime_node.ProcessId(), runtime_node.ThreadId()});

  output_file_stream_ << string_format(
      std::string(
          R"JSON(
  { 
    "name": "launch", "id": %d, "pid": %lld, "tid": "%lld(C++)",
    "ts": %lld, 
    "ph": "s", "cat": "async"
  },
  )JSON"),
      runtime_node.CorrelationId(), runtime_node.ProcessId(),
      runtime_node.ThreadId(),
      nsToUs((runtime_node.StartNs() + runtime_node.EndNs()) >> 1));
  pid_tid_set_.insert({runtime_node.ProcessId(), runtime_node.ThreadId()});
}

void ChromeTracingLogger::LogDeviceTraceEventNode(
    const DeviceTraceEventNode& device_node) {
  if (!output_file_stream_) {
    return;
  }
  switch (device_node.Type()) {
    case TracerEventType::Kernel:
      HandleTypeKernel(device_node);
      break;
    case TracerEventType::Memcpy:
      HandleTypeMemcpy(device_node);
      break;
    case TracerEventType::Memset:
      HandleTypeMemset(device_node);
    default:
      break;
  }
  if (nsToUs(device_node.Duration()) == 0) {
    output_file_stream_ << string_format(
        std::string(
            R"JSON(
  { 
    "name": "launch", "id": %d, "pid": %lld, "tid": %lld,
    "ts": %lld, 
    "ph": "f", "cat": "async"
  },
  )JSON"),
        device_node.CorrelationId(), device_node.DeviceId(),
        device_node.StreamId(), nsToUs(device_node.StartNs()));
    deviceid_streamid_set_.insert(
        {device_node.DeviceId(), device_node.StreamId()});
  } else {
    output_file_stream_ << string_format(
        std::string(
            R"JSON(
  { 
    "name": "launch", "id": %d, "pid": %lld, "tid": %lld,
    "ts": %lld, 
    "ph": "f", "cat": "async", "bp": "e"
  },
  )JSON"),
        device_node.CorrelationId(), device_node.DeviceId(),
        device_node.StreamId(),
        nsToUs((device_node.StartNs() + device_node.EndNs()) >> 1));
    deviceid_streamid_set_.insert(
        {device_node.DeviceId(), device_node.StreamId()});
  }
}

void ChromeTracingLogger::HandleTypeKernel(
    const DeviceTraceEventNode& device_node) {
  KernelEventInfo kernel_info = device_node.KernelInfo();
  float blocks_per_sm = 0.0;
  float warps_per_sm = 0.0;
  float occupancy = 0.0;
#if defined(PADDLE_WITH_CUPTI)
  constexpr int threads_per_warp = 32;
  const gpuDeviceProp& device_property =
      GetDeviceProperties(device_node.DeviceId());
  blocks_per_sm = static_cast<float>(kernel_info.grid_x * kernel_info.grid_y *
                                     kernel_info.grid_z) /
                  device_property.multiProcessorCount;
  warps_per_sm = blocks_per_sm * (kernel_info.block_x * kernel_info.block_y *
                                  kernel_info.block_z) /
                 threads_per_warp;
  occupancy = CalculateEstOccupancy(
      device_node.DeviceId(), kernel_info.registers_per_thread,
      kernel_info.static_shared_memory, kernel_info.dynamic_shared_memory,
      kernel_info.block_x, kernel_info.block_y, kernel_info.block_z,
      blocks_per_sm);
#endif

  output_file_stream_ << string_format(
      std::string(
          R"JSON(
  { 
    "name": "%s", "pid": %lld, "tid": %lld,
    "ts": %lld, "dur": %lld,
    "ph": "X", "cat": "%s", 
    "args": {
      "start_ns": %lld,
      "end_ns": %lld,
      "device": %d, "context": %d,
      "stream": %d, "correlation id": %d,
      "registers per thread": %d,
      "shared memory": %d,
      "blocks per SM": %f,
      "warps per SM": %f,
      "grid": [%d, %d, %d],
      "block": [%d, %d, %d],
      "theoretical achieved occupancy %%": %f
    }
  },
  )JSON"),
      device_node.Name().c_str(), device_node.DeviceId(),
      device_node.StreamId(), nsToUs(device_node.StartNs()),
      nsToUs(device_node.Duration()),
      categary_name_[static_cast<int>(device_node.Type())],
      device_node.StartNs(), device_node.EndNs(), device_node.DeviceId(),
      device_node.ContextId(), device_node.StreamId(),
      device_node.CorrelationId(), kernel_info.registers_per_thread,
      kernel_info.static_shared_memory + kernel_info.dynamic_shared_memory,
      blocks_per_sm, warps_per_sm, kernel_info.grid_x, kernel_info.grid_y,
      kernel_info.grid_z, kernel_info.block_x, kernel_info.block_y,
      kernel_info.block_z, occupancy * 100);
}

void ChromeTracingLogger::HandleTypeMemcpy(
    const DeviceTraceEventNode& device_node) {
  MemcpyEventInfo memcpy_info = device_node.MemcpyInfo();
  float memory_bandwidth = 0;
  if (device_node.Duration() > 0) {
    memory_bandwidth = memcpy_info.num_bytes * 1.0 / device_node.Duration();
  }
  output_file_stream_ << string_format(
      std::string(
          R"JSON(
  {
    "name": "%s", "pid": %lld, "tid": %lld,
    "ts": %lld, "dur": %lld,
    "ph": "X", "cat": "%s", 
    "args": {
      "start_ns": %lld,
      "end_ns": %lld,
      "stream": %d, "correlation id": %d,
      "bytes": %d, "memory bandwidth (GB/s)": %f
    }
  },
  )JSON"),
      device_node.Name().c_str(), device_node.DeviceId(),
      device_node.StreamId(), nsToUs(device_node.StartNs()),
      nsToUs(device_node.Duration()),
      categary_name_[static_cast<int>(device_node.Type())],
      device_node.StartNs(), device_node.EndNs(), device_node.StreamId(),
      device_node.CorrelationId(), memcpy_info.num_bytes, memory_bandwidth);
}

void ChromeTracingLogger::HandleTypeMemset(
    const DeviceTraceEventNode& device_node) {
  MemsetEventInfo memset_info = device_node.MemsetInfo();
  output_file_stream_ << string_format(
      std::string(
          R"JSON(
  {
    "name": "%s", "pid": %lld, "tid": %lld,
    "ts": %lld, "dur": %lld,
    "ph": "X", "cat": "%s", 
    "args": {
      "start_ns": %lld,
      "end_ns": %lld,
      "device": %d, "context": %d,
      "stream": %d, "correlation id": %d,
      "bytes": %d, "value": %d
    }
  },
  )JSON"),
      device_node.Name().c_str(), device_node.DeviceId(),
      device_node.StreamId(), nsToUs(device_node.StartNs()),
      nsToUs(device_node.Duration()),
      categary_name_[static_cast<int>(device_node.Type())],
      device_node.StartNs(), device_node.EndNs(), device_node.DeviceId(),
      device_node.ContextId(), device_node.StreamId(),
      device_node.CorrelationId(), memset_info.num_bytes, memset_info.value);
}

void ChromeTracingLogger::StartLog() {
  output_file_stream_ << string_format(std::string(
                                           R"JSON(
  { 
    "schemaVersion": "%s",
    "displayTimeUnit": "ms",
    "span_indx": "%d",
  )JSON"),
                                       kSchemaVersion, span_indx++);
// add device property information
#if defined(PADDLE_WITH_CUDA)
  output_file_stream_ << std::string(R"JSON(
    "deviceProperties": [
  )JSON");
  std::vector<int> device_ids = GetSelectedDevices();
  for (auto index = 0u; index < device_ids.size() - 1; index++) {
    const gpuDeviceProp& device_property =
        GetDeviceProperties(device_ids[index]);
    output_file_stream_ << string_format(
        std::string(
            R"JSON(
    {
       "id": %d, "name": "%s", "totalGlobalMem": %u,
      "computeMajor": %d, "computeMinor": %d,
      "maxThreadsPerBlock": %d, "maxThreadsPerMultiprocessor": %d,
      "regsPerBlock": %d, "regsPerMultiprocessor": %d, "warpSize": %d,
      "sharedMemPerBlock": %d, "sharedMemPerMultiprocessor": %d,
      "smCount": %d, "sharedMemPerBlockOptin": %d
    },
  )JSON"),
        device_ids[index], device_property.name, device_property.totalGlobalMem,
        device_property.major, device_property.minor,
        device_property.maxThreadsPerBlock,
        device_property.maxThreadsPerMultiProcessor,
        device_property.regsPerBlock, device_property.regsPerMultiprocessor,
        device_property.warpSize, device_property.sharedMemPerBlock,
        device_property.sharedMemPerMultiprocessor,
        device_property.multiProcessorCount,
        device_property.sharedMemPerBlockOptin);
  }
  if (device_ids.size() > 0) {
    const gpuDeviceProp& device_property =
        GetDeviceProperties(device_ids[device_ids.size() - 1]);
    output_file_stream_ << string_format(
        std::string(
            R"JSON(
    {
       "id": %d, "name": "%s", "totalGlobalMem": %u,
      "computeMajor": %d, "computeMinor": %d,
      "maxThreadsPerBlock": %d, "maxThreadsPerMultiprocessor": %d,
      "regsPerBlock": %d, "regsPerMultiprocessor": %d, "warpSize": %d,
      "sharedMemPerBlock": %d, "sharedMemPerMultiprocessor": %d,
      "smCount": %d, "sharedMemPerBlockOptin": %d
    }],
  )JSON"),
        device_ids[device_ids.size() - 1], device_property.name,
        device_property.totalGlobalMem, device_property.major,
        device_property.minor, device_property.maxThreadsPerBlock,
        device_property.maxThreadsPerMultiProcessor,
        device_property.regsPerBlock, device_property.regsPerMultiprocessor,
        device_property.warpSize, device_property.sharedMemPerBlock,
        device_property.sharedMemPerMultiprocessor,
        device_property.multiProcessorCount,
        device_property.sharedMemPerBlockOptin);
  }
#endif

  output_file_stream_ << std::string(
      R"JSON(
    "traceEvents": [
  )JSON");
}

void ChromeTracingLogger::LogMetaInfo(
    const std::unordered_map<std::string, std::string> extra_info) {
  RefineDisplayName(extra_info);
  output_file_stream_ << std::string(
      R"JSON(
  {}
  ],
  )JSON");
  output_file_stream_ << std::string(R"JSON(
  "ExtraInfo": {)JSON");
  size_t count = extra_info.size();
  for (const auto& kv : extra_info) {
    if (count > 1) {
      output_file_stream_ << string_format(std::string(R"JSON(
     "%s": "%s",
   )JSON"),
                                           kv.first.c_str(), kv.second.c_str());
    } else {
      output_file_stream_ << string_format(std::string(R"JSON(
     "%s": "%s"
   )JSON"),
                                           kv.first.c_str(), kv.second.c_str());
    }
    count--;
  }
  output_file_stream_ << std::string(R"JSON(
  })JSON");
}

void ChromeTracingLogger::RefineDisplayName(
    std::unordered_map<std::string, std::string> extra_info) {
  for (auto it = pid_tid_set_.begin(); it != pid_tid_set_.end(); ++it) {
    output_file_stream_ << string_format(
        std::string(
            R"JSON(
  {
    "name": "process_name", "pid": %lld, "tid": "%lld(Python)",
    "ph": "M", 
    "args": {
      "name": "Process %lld (CPU)"
    }
  },
  {
    "name": "process_name", "pid": %lld, "tid": "%lld(C++)",
    "ph": "M", 
    "args": {
      "name": "Process %lld (CPU)"
    }
  },
   {
    "name": "thread_name", "pid": %lld, "tid": "%lld(Python)",
    "ph": "M", 
    "args": {
      "name": "thread %lld:%s(Python)"
    }
  },
  {
    "name": "thread_name", "pid": %lld, "tid": "%lld(C++)",
    "ph": "M", 
    "args": {
      "name": "thread %lld:%s(C++)"
    }
  },
  {
    "name": "process_sort_index", "pid": %lld, "tid": %lld,
    "ph": "M", 
    "args": {
      "sort_index": %lld
    }
  },  
  {
    "name": "thread_sort_index", "pid": %lld, "tid": "%lld(Python)",
    "ph": "M", 
    "args": {
      "sort_index": %lld
    }
  },
  {
    "name": "thread_sort_index", "pid": %lld, "tid": "%lld(C++)",
    "ph": "M", 
    "args": {
      "sort_index": %lld
    }
  },
  )JSON"),
        (*it).first, (*it).second, (*it).first, (*it).first, (*it).second,
        (*it).first, (*it).first, (*it).second, (*it).second,
        extra_info[string_format(std::string("%lld"), (*it).second)].c_str(),
        (*it).first, (*it).second, (*it).second,
        extra_info[string_format(std::string("%lld"), (*it).second)].c_str(),
        (*it).first, (*it).second, (*it).first, (*it).first, (*it).second,
        (*it).second * 2, (*it).first, (*it).second, (*it).second * 2 + 1);
  }

  for (auto it = deviceid_streamid_set_.begin();
       it != deviceid_streamid_set_.end(); ++it) {
    output_file_stream_ << string_format(
        std::string(
            R"JSON(
  {
    "name": "process_name", "pid": %lld, "tid": %lld,
    "ph": "M", 
    "args": {
      "name": "Deivce %lld (GPU)"
    }
  },
   {
    "name": "thread_name", "pid": %lld, "tid": %lld,
    "ph": "M", 
    "args": {
      "name": "stream %lld"
    }
  },
  {
    "name": "process_sort_index", "pid": %lld, "tid": %lld,
    "ph": "M", 
    "args": {
      "sort_index": %lld
    }
  },  
  {
    "name": "thread_sort_index", "pid": %lld, "tid": %lld,
    "ph": "M", 
    "args": {
      "sort_index": %lld
    }
  },  
  )JSON"),
        (*it).first, (*it).second, (*it).first, (*it).first, (*it).second,
        (*it).second, (*it).first, (*it).second, (*it).first + 0x10000000,
        (*it).first, (*it).second, (*it).second);
  }
}

void ChromeTracingLogger::EndLog() {
  output_file_stream_ << std::string(
      R"JSON(
  }
  )JSON");
}

}  // namespace platform
}  // namespace paddle
