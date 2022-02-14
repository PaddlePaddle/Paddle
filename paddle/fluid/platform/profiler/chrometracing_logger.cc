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
#include "paddle/fluid/platform/os_info.h"
#include "paddle/fluid/platform/profiler/chrometracing_logger.h"
#include "paddle/fluid/platform/profiler/event_node.h"

namespace paddle {
namespace platform {

static const char* kSchemaVersion = "1.0.0";
static const char* kDefaultFilename = "pid_%s_time_%s.paddle_trace.json";
static uint32_t num_span = 0;

static int64_t nsToUs(int64_t ns) { return ns / 1000; }

template <typename... Args>
std::string string_format(const std::string& format, Args... args) {
  int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) +
               1;  // Extra space for '\0'
  PADDLE_ENFORCE_GE(size_s, 0, platform::errors::Fatal(
                                   "Error during profiler data formatting."));
  auto size = static_cast<size_t>(size_s);
  auto buf = std::make_unique<char[]>(size);
  std::snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(), size - 1);  // exclude the '\0'
}

std::string GetStringFormatLocalTime() {
  std::time_t rawtime;
  std::tm* timeinfo;
  char buf[100];
  std::time(&rawtime);
  timeinfo = std::localtime(&rawtime);
  std::strftime(buf, 100, "%F-%X", timeinfo);
  return std::string(buf);
}

static std::string DefaultFileName() {
  auto pid = GetProcessId();
  return string_format(std::string(kDefaultFilename), pid,
                       GetStringFormatLocalTime().c_str());
}

const char* ChromeTracingLogger::categary_name_[] = {
    "operator", "dataloader", "profile_step", "cuda_runtime", "kernel",
    "memcpy",   "memset",     "user_defined", "others"};

void ChromeTracingLogger::OpenFile() {
  output_file_stream_.open(filename_,
                           std::ofstream::out | std::ofstream::trunc);
  if (!output_file_stream_) {
    VLOG(2) << "Unable to open file for writing profiling data." << std::endl;
  } else {
    VLOG(0) << "writing profiling data to " << filename_ << std::endl;
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
  output_file_stream_ << string_format(
      std::string(
          R"JSON(
  { 
    "name": "%s", "pid": %lld, "tid": %lld,
    "ts": %lld, "dur": %lld,
    "ph": "X", "cat": "%s", 
    "args": {
      
    }
  },
  )JSON"),
      host_node.Name().c_str(), host_node.ProcessId(), host_node.ThreadId(),
      nsToUs(host_node.StartNs()), nsToUs(host_node.Duration()),
      categary_name_[static_cast<int>(host_node.Type())]);
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
    "name": "%s", "pid": %lld, "tid": %lld,
    "ts": %lld, "dur": %lld,
    "ph": "X", "cat": "%s", 
    "args": {
      "correlation id": %d
    }
  },
  )JSON"),
      runtime_node.Name().c_str(), runtime_node.ProcessId(),
      runtime_node.ThreadId(), nsToUs(runtime_node.StartNs()),
      nsToUs(runtime_node.Duration()),
      categary_name_[static_cast<int>(runtime_node.Type())],
      runtime_node.CorrelationId());
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
}

void ChromeTracingLogger::HandleTypeKernel(
    const DeviceTraceEventNode& device_node) {
  KernelEventInfo kernel_info = device_node.KernelInfo();
  float blocks_per_sm = 0.0;
  float warps_per_sm = 0.0;
  float occupancy = 0.0;
#if defined(PADDLE_WITH_CUDA)
  constexpr int threads_per_warp = 32;
  const gpuDeviceProp& device_property =
      GetDeviceProperties(device_node.DeviceId());
  blocks_per_sm =
      (kernel_info.grid_x * kernel_info.grid_y * kernel_info.grid_z) /
      device_property.multiProcessorCount;
  warps_per_sm = blocks_per_sm * (kernel_info.block_x * kernel_info.block_y *
                                  kernel_info.block_z) /
                 threads_per_warp;
#endif

  output_file_stream_ << string_format(
      std::string(
          R"JSON(
  { 
    "name": "%s", "pid": %lld, "tid": %lld,
    "ts": %lld, "dur": %lld,
    "ph": "X", "cat": "%s", 
    "args": {
      "device": %d, "context": %d,
      "stream": %d, "correlation id": %d,
      "registers per thread": %d,
      "shared memory": %f,
      "blocks per SM": %f,
      "warps per SM": %f,
      "grid": [%d, %d, %d],
      "block": [%d, %d, %d],
      "est. achieved occupancy %": %f
    }
  },
  )JSON"),
      device_node.Name().c_str(), device_node.DeviceId(),
      device_node.StreamId(), nsToUs(device_node.StartNs()),
      nsToUs(device_node.Duration()),
      categary_name_[static_cast<int>(device_node.Type())],
      device_node.DeviceId(), device_node.ContextId(), device_node.StreamId(),
      device_node.CorrelationId(), kernel_info.registers_per_thread,
      kernel_info.static_shared_memory + kernel_info.dynamic_shared_memory,
      blocks_per_sm, warps_per_sm, kernel_info.grid_x, kernel_info.grid_y,
      kernel_info.grid_z, kernel_info.block_x, kernel_info.block_y,
      kernel_info.block_z, occupancy);
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
      "stream": %d, "correlation id": %d,
      "bytes": %d, "memory bandwidth (GB/s)": %f
    }
  },
  )JSON"),
      device_node.Name().c_str(), device_node.DeviceId(),
      device_node.StreamId(), nsToUs(device_node.StartNs()),
      nsToUs(device_node.Duration()),
      categary_name_[static_cast<int>(device_node.Type())],
      device_node.StreamId(), device_node.CorrelationId(),
      memcpy_info.num_bytes, memory_bandwidth);
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
      device_node.DeviceId(), device_node.ContextId(), device_node.StreamId(),
      device_node.CorrelationId(), memset_info.num_bytes, memset_info.value);
}

void ChromeTracingLogger::StartLog() {
  output_file_stream_ << string_format(std::string(
                                           R"JSON(
  { 
    "schemaVersion": "%s",
    "displayTimeUnit": "us",
    "SpanNumber": "%d",
  )JSON"),
                                       kSchemaVersion, num_span);
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

void ChromeTracingLogger::EndLog() {
  output_file_stream_ << std::string(
      R"JSON(
  {}
  ]
  }
  )JSON");
}

}  // namespace platform
}  // namespace paddle
