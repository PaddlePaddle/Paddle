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

#include <map>
#include <unordered_map>

#include "paddle/fluid/platform/profiler/dump/nodetree.pb.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#include "paddle/phi/core/platform/profiler/output_logger.h"

namespace paddle {
namespace platform {

// Dump a NodeTrees into a protobuf file.
// A SerializationLogger object can only dump a NodeTrees object,
// creates a file in the constructor and closes the file in the destructor.
// Should only call LogNodeTrees and LogMetaInfo.
class SerializationLogger : public BaseLogger {
 public:
  explicit SerializationLogger(const std::string& filename);
  explicit SerializationLogger(const char* filename);
  ~SerializationLogger();
  std::string filename() { return filename_; }
  void LogDeviceTraceEventNode(const DeviceTraceEventNode&) override;
  void LogHostTraceEventNode(const HostTraceEventNode&) override;
  void LogRuntimeTraceEventNode(const CudaRuntimeTraceEventNode&) override;
  void LogNodeTrees(const NodeTrees&) override;
  void LogExtraInfo(const std::unordered_map<std::string, std::string>);
  void LogMemTraceEventNode(const MemTraceEventNode&) override;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  void LogDeviceProperty(
      const std::map<uint32_t, gpuDeviceProp>& device_property_map);
#endif
  void LogMetaInfo(const std::string& version, uint32_t span_indx);

 private:
  void OpenFile();
  void HandleTypeKernel(const DeviceTraceEventNode&);
  void HandleTypeMemset(const DeviceTraceEventNode&);
  void HandleTypeMemcpy(const DeviceTraceEventNode&);

  std::string filename_;
  std::ofstream output_file_stream_;
  NodeTreesProto* node_trees_proto_ = nullptr;
  ThreadNodeTreeProto* current_thread_node_tree_proto_ = nullptr;
  HostTraceEventNodeProto* current_host_trace_event_node_proto_ = nullptr;
  CudaRuntimeTraceEventNodeProto* current_runtime_trace_event_node_proto_ =
      nullptr;
  DeviceTraceEventNodeProto* current_device_trace_event_node_proto_ = nullptr;
  MemTraceEventNodeProto* current_mem_trace_event_node_proto_ = nullptr;
  OperatorSupplementEventNodeProto* current_op_supplement_event_node_proto_ =
      nullptr;
};

}  // namespace platform
}  // namespace paddle
