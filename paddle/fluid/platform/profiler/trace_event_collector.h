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

#include <list>

namespace paddle {
namespace platform {

class TraceEventCollector {
 public:
  void AddHostRecord(HostRecord&& record) { host_records_.push_back(record); }

  void AddRuntimeRecord(RuntimeRecord&& record) {
    runtime_records_.push_back(record);
  }

  void AddDeviceRecord(DeviceRecord&& record) {
    device_records_.push_back(record);
  }

 private:
  std::list<HostRecord> host_records_;
  std::list<RuntimeRecord> runtime_records_;
  std::list<DeviceRecord> device_records_;
};

}  // namespace platform
}  // namespace paddle
