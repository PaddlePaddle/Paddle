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

#include <fstream>
#include <ostream>

namespace paddle {
namespace platform {
class DeviceRecordNode;       // forward declaration
class HostRecordNode;         // forward declaration
class CudaRuntimeRecordNode;  // forward declaration

class BaseLogger {
 public:
  BaseLogger() {}
  virtual ~BaseLogger() {}
  virtual void LogDeviceRecordNode(DeviceRecordNode&) {}
  virtual void LogHostRecordNode(HostRecordNode&) {}
  virtual void LogRuntimeRecordNode(CudaRuntimeRecordNode&) {}
  virtual void LogMetaInfo() {}
};

class ChromeTracingLogger : public BaseLogger {
 public:
  explicit ChromeTracingLogger(const std::string& filename);
  explicit ChromeTracingLogger(const char* filename);
  ~ChromeTracingLogger();
  std::string filename() { return filename_; }
  void LogDeviceRecordNode(DeviceRecordNode&) override;
  void LogHostRecordNode(HostRecordNode&) override;
  void LogRuntimeRecordNode(CudaRuntimeRecordNode&) override;
  void LogMetaInfo();

 private:
  void OpenFile();
  void HandleTypeKernel(DeviceRecordNode&);
  void HandleTypeMemset(DeviceRecordNode&);
  void HandleTypeMemcpy(DeviceRecordNode&);
  void StartLog();
  void EndLog();
  std::string filename_;
  std::ofstream output_file_stream_;
  static const char* categary_name_[];
};

}  // namespace platform
}  // namespace paddle
