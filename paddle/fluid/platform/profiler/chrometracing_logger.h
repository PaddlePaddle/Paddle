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

#include <cstdint>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>

#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#include "paddle/phi/core/platform/profiler/output_logger.h"

namespace paddle {
namespace platform {

// Dump a NodeTrees into a chrome tracing file.
// A ChromeTracingLogger object can only dump a NodeTrees object,
// creates a file in the constructor and closes the file in the destructor.
// should only call LogNodeTrees and LogMetaInfo in order.
class ChromeTracingLogger : public BaseLogger {
 public:
  explicit ChromeTracingLogger(const std::string& filename);
  explicit ChromeTracingLogger(const char* filename);
  ~ChromeTracingLogger();
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
  void StartLog();
  void EndLog();
  void RefineDisplayName(std::unordered_map<std::string, std::string>);
  std::string filename_;
  std::ofstream output_file_stream_;
  static const char* category_name_[];
  std::set<std::pair<uint64_t, uint64_t>> pid_tid_set_;
  std::set<std::pair<uint64_t, uint64_t>> deviceid_streamid_set_;
  uint64_t start_time_;
};

}  // namespace platform
}  // namespace paddle
