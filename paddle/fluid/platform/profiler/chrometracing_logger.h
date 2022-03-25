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

#include <set>
#include <unordered_map>
#include <utility>
#include "paddle/fluid/platform/profiler/output_logger.h"

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
  void LogMetaInfo(const std::unordered_map<std::string, std::string>);

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
  static const char* categary_name_[];
  std::set<std::pair<uint64_t, uint64_t>> pid_tid_set_;
  std::set<std::pair<uint64_t, uint64_t>> deviceid_streamid_set_;
};

}  // namespace platform
}  // namespace paddle
