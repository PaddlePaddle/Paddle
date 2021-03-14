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

#include <chrono>  // NOLINT
#include <string>

#include "paddle/fluid/platform/dynload/cupti.h"
#include "paddle/fluid/platform/event.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/port.h"
#include "paddle/fluid/platform/profiler.pb.h"

namespace paddle {
namespace platform {

///////////////////////
// WARN: Under Development. Don't depend on it yet.
//////////////////////
class Event;

inline uint64_t PosixInNsec() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return 1000 * (static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec);
}

// DeviceTracer performs the following tasks:
// 1. Register cuda callbacks for various events: kernel, memcpy, etc.
// 2. Collect cuda statistics: start/end ts, memory, etc.
// 3. Generate a protobuf for further analysis.
class DeviceTracer {
 public:
  struct KernelRecord {
    std::string name;
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

  struct MemInfoRecord {
    uint64_t start_ns;
    uint64_t end_ns;
    size_t bytes;
    Place place;
    int64_t thread_id;
    std::string alloc_in;
    std::string free_in;
  };

  struct ActiveKindRecord {
    std::string name;
    uint64_t start_ns;
    uint64_t end_ns;
    int64_t device_id;
    int64_t thread_id;
    uint32_t correlation_id;
  };

  virtual ~DeviceTracer() {}
  // Needs to be called once before use.
  virtual void Enable() = 0;
  // Needs to be called once after use.
  virtual void Disable() = 0;
  // Needs to be called once before reuse.
  virtual void Reset() = 0;

  // Add a pair to correlate internal cuda id with high level
  // annotation event(with string). So cuda statistics can be represented by
  // human-readable annotations.
  virtual void AddAnnotation(uint32_t id, Event* event) = 0;

  virtual void AddMemRecords(const std::string& name, uint64_t start_ns,
                             uint64_t end_ns, int64_t device_id,
                             int64_t stream_id, uint32_t correlation_id,
                             uint64_t bytes) = 0;

  virtual void AddCPURecords(const std::string& anno, uint64_t start_ns,
                             uint64_t end_ns, int64_t device_id,
                             int64_t thread_id) = 0;
  virtual void AddActiveKindRecords(const std::string& anno, uint64_t start_ns,
                                    uint64_t end_ns, int64_t device_id,
                                    int64_t thread_id,
                                    uint32_t correlation_id) = 0;

  virtual void AddMemInfoRecord(uint64_t start_ns, uint64_t end_ns,
                                size_t bytes, const Place& place,
                                const std::string& alloc_in,
                                const std::string& free_in,
                                int64_t thread_id) = 0;

  // Add a cuda kernel stats. `correlation_id` will be mapped to annotation
  // added before for human readability.
  virtual void AddKernelRecords(std::string name, uint64_t start, uint64_t end,
                                int64_t device_id, int64_t stream_id,
                                uint32_t correlation_id) = 0;

  // Generate a proto after done (Disabled).
  virtual proto::Profile GenProfile(const std::string& profile_path) = 0;

  // generate kernel elapsed time into Event
  virtual void GenEventKernelCudaElapsedTime() = 0;

  virtual bool IsEnabled() = 0;
};

// Get a DeviceTracer.
DeviceTracer* GetDeviceTracer();

// Set a name for the cuda kernel operation being launched by the thread.
void SetCurAnnotation(Event* event);
// Clear the name after the operation is done.
void ClearCurAnnotation();
// Current name of the operation being run in the thread.
std::string CurAnnotationName();
Event* CurAnnotation();

void SetCurBlock(int block_id);
void ClearCurBlock();
int BlockDepth();

// Set current thread id, so we can map the system thread id to thread id.
void RecoreCurThreadId(int32_t id);
int32_t GetThreadIdFromSystemThreadId(uint32_t id);
}  // namespace platform
}  // namespace paddle
