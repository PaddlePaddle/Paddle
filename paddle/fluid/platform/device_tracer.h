/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include <string>

#include "paddle/fluid/platform/dynload/cupti.h"
#include "paddle/fluid/platform/profiler.pb.h"

namespace paddle {
namespace platform {

///////////////////////
// WARN: Under Development. Don't depend on it yet.
//////////////////////

// DeviceTracer performs the following tasks:
// 1. Register cuda callbacks for various events: kernel, memcpy, etc.
// 2. Collect cuda statistics: start/end ts, memory, etc.
// 3. Generate a protobuf for further analysis.
class DeviceTracer {
 public:
  struct KernelRecord {
    uint64_t start_ns;
    uint64_t end_ns;
    int64_t device_id;
    int64_t stream_id;
    uint32_t correlation_id;
  };
  struct CPURecord {
    std::string name;
    uint64_t start_ns;
    uint64_t end_ns;
    int64_t device_id;
    int64_t thread_id;
  };
  struct MemRecord {
    std::string name;
    uint64_t start_ns;
    uint64_t end_ns;
    int64_t device_id;
    int64_t stream_id;
    uint32_t correlation_id;
    uint64_t bytes;
  };

  virtual ~DeviceTracer() {}
  // Needs to be called once before use.
  virtual void Enable() = 0;
  // Needs to be called once after use.
  virtual void Disable() = 0;

  // Add a pair to correlate internal cuda id with high level
  // annotation (string). So cuda statistics can be represented by
  // human-readable annotations.
  virtual void AddAnnotation(uint64_t id, const std::string& anno) = 0;

  virtual void AddMemRecords(const std::string& name, uint64_t start_ns,
                             uint64_t end_ns, int64_t device_id,
                             int64_t stream_id, uint32_t correlation_id,
                             uint64_t bytes) = 0;

  virtual void AddCPURecords(const std::string& anno, uint64_t start_ns,
                             uint64_t end_ns, int64_t device_id,
                             int64_t thread_id) = 0;

  // Add a cuda kernel stats. `correlation_id` will be mapped to annotation
  // added before for human readability.
  virtual void AddKernelRecords(uint64_t start, uint64_t end, int64_t device_id,
                                int64_t stream_id, uint32_t correlation_id) = 0;

  // Generate a proto after done (Disabled).
  virtual proto::Profile GenProfile(const std::string& profile_path) = 0;

  virtual bool IsEnabled() = 0;
};

// Get a DeviceTracer.
DeviceTracer* GetDeviceTracer();

// Set a name for the cuda kernel operation being launched by the thread.
void SetCurAnnotation(const std::string& anno);
// Clear the name after the operation is done.
void ClearCurAnnotation();
// Current name of the operation being run in the thread.
std::string CurAnnotation();

void SetCurBlock(int block_id);
void ClearCurBlock();
int BlockDepth();

void SetCurThread(int thread_id);
void ClearCurThread();
int CurThread();
}  // namespace platform
}  // namespace paddle
